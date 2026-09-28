"""F1 benchmark environment: a damped nonlinear pendulum.

The environment exposes a latent state x = [theta, omega] and a scalar
angle sensor.  It is deliberately independent of the EKF (F2) so that the
same datasets can be reused by multiple estimators later.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PendulumConfig:
    """Physical parameters and initial condition for the benchmark."""

    gravity: float = 9.81
    length: float = 1.0
    damping_start: float = 0.08
    damping_end: float | None = None
    initial_angle: float = np.deg2rad(35.0)
    initial_velocity: float = 0.0


@dataclass(frozen=True)
class Scenario:
    """Measurement and environmental conditions for one F1 experiment."""

    name: str
    description: str
    duration: float = 20.0
    dt: float = 0.01
    measurement_stride: int = 1
    measurement_noise_std: float = np.deg2rad(1.0)
    damping_end: float | None = None
    sensor_bias_rate: float = 0.0
    seed: int = 0


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="nominal",
        description="Regular angle measurements with low Gaussian sensor noise.",
        seed=101,
    ),
    Scenario(
        name="high_noise",
        description="Regular measurements with four times the nominal sensor noise.",
        measurement_noise_std=np.deg2rad(4.0),
        seed=202,
    ),
    Scenario(
        name="sparse_measurements",
        description="A measurement is available only once every ten integration steps.",
        measurement_stride=10,
        seed=303,
    ),
    Scenario(
        name="parameter_and_sensor_drift",
        description=(
            "Damping increases linearly during the run and the angle sensor has "
            "a slowly changing positive bias."
        ),
        damping_end=0.16,
        sensor_bias_rate=np.deg2rad(0.04),
        seed=404,
    ),
)


def _damping_at(time: float, config: PendulumConfig, duration: float) -> float:
    """Return the damping coefficient at ``time`` for the selected run."""

    end = config.damping_end if config.damping_end is not None else config.damping_start
    fraction = np.clip(time / max(duration, np.finfo(float).eps), 0.0, 1.0)
    return float(config.damping_start + fraction * (end - config.damping_start))


def _derivative(
    time: float,
    state: np.ndarray,
    config: PendulumConfig,
    duration: float,
) -> np.ndarray:
    theta, omega = state
    damping = _damping_at(time, config, duration)
    return np.array(
        [
            omega,
            -(config.gravity / config.length) * np.sin(theta) - damping * omega,
        ],
        dtype=float,
    )


def _rk4_step(
    time: float,
    state: np.ndarray,
    dt: float,
    config: PendulumConfig,
    duration: float,
) -> np.ndarray:
    """Advance the nonlinear ODE one step with fourth-order Runge--Kutta."""

    k1 = _derivative(time, state, config, duration)
    k2 = _derivative(time + dt / 2, state + dt * k1 / 2, config, duration)
    k3 = _derivative(time + dt / 2, state + dt * k2 / 2, config, duration)
    k4 = _derivative(time + dt, state + dt * k3, config, duration)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_truth(config: PendulumConfig, duration: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate and return ``(time, state, damping)`` for one trajectory."""

    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration):
        raise ValueError("duration must be an integer multiple of dt")

    time = np.arange(steps + 1, dtype=float) * dt
    states = np.empty((steps + 1, 2), dtype=float)
    states[0] = [config.initial_angle, config.initial_velocity]
    for index in range(steps):
        states[index + 1] = _rk4_step(time[index], states[index], dt, config, duration)

    damping = np.array([_damping_at(t, config, duration) for t in time])
    return time, states, damping


def make_observations(
    time: np.ndarray,
    states: np.ndarray,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sparse noisy angle measurements.

    Returns ``(observations, observed_mask, measurement_times, measurement_values)``.
    ``observations`` is aligned to the simulation grid and contains NaN where
    no sensor reading was taken.
    """

    if scenario.measurement_stride < 1:
        raise ValueError("measurement_stride must be at least one")
    if scenario.measurement_noise_std < 0:
        raise ValueError("measurement_noise_std cannot be negative")

    rng = np.random.default_rng(scenario.seed)
    observed_mask = np.zeros(len(time), dtype=bool)
    observed_mask[:: scenario.measurement_stride] = True
    bias = scenario.sensor_bias_rate * time
    noise = rng.normal(0.0, scenario.measurement_noise_std, size=len(time))
    observations = np.full(len(time), np.nan, dtype=float)
    observations[observed_mask] = states[observed_mask, 0] + bias[observed_mask] + noise[observed_mask]
    return (
        observations,
        observed_mask,
        time[observed_mask],
        observations[observed_mask],
    )


def generate_dataset(
    scenario: Scenario,
    config: PendulumConfig | None = None,
) -> dict[str, np.ndarray | str]:
    """Generate one complete F1 dataset and its metadata."""

    config = config or PendulumConfig()
    effective_config = replace(
        config,
        damping_end=(
            scenario.damping_end
            if scenario.damping_end is not None
            else config.damping_end
        ),
    )
    time, states, damping = simulate_truth(effective_config, scenario.duration, scenario.dt)
    observations, observed_mask, measurement_times, measurement_values = make_observations(
        time, states, scenario
    )
    return {
        "time": time,
        "true_state": states,
        "damping": damping,
        "sensor_bias": scenario.sensor_bias_rate * time,
        "observations": observations,
        "observed_mask": observed_mask,
        "measurement_times": measurement_times,
        "measurement_values": measurement_values,
        "scenario": scenario.name,
    }


def write_benchmark(
    output_dir: Path,
    scenarios: Iterable[Scenario] = SCENARIOS,
    config: PendulumConfig | None = None,
) -> Path:
    """Write compressed datasets plus a machine-readable manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or PendulumConfig()
    manifest: dict[str, object] = {
        "system": "damped_nonlinear_pendulum",
        "state_definition": ["theta_rad", "omega_rad_per_s"],
        "observation_definition": "angle_rad + sensor_bias_rad + Gaussian_noise_rad",
        "integrator": "fixed-step RK4",
        "physical_parameters": asdict(config),
        "datasets": [],
    }

    for scenario in scenarios:
        dataset = generate_dataset(scenario, config)
        filename = f"pendulum_{scenario.name}.npz"
        np.savez_compressed(
            output_dir / filename,
            **{key: value for key, value in dataset.items() if isinstance(value, np.ndarray)},
        )
        manifest["datasets"].append(
            {
                **asdict(scenario),
                "file": filename,
                "samples": int(len(dataset["time"])),
                "measurements": int(np.sum(dataset["observed_mask"])),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the F1 damped-pendulum benchmark.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/f1_pendulum"),
        help="directory for .npz datasets and manifest.json",
    )
    args = parser.parse_args()
    manifest_path = write_benchmark(args.output)
    print(f"Wrote F1 benchmark manifest: {manifest_path}")


if __name__ == "__main__":
    main()
