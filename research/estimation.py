"""Classical EKF and an idealized Ramsey spin sensor.

The magnetic field B(theta)=G*L*sin(theta) assumes a uniform transverse
field gradient. It is a simplified sensing hypothesis, not an NV hardware
model. All sensor controls depend on the filter prior, never on truth.
"""
from dataclasses import dataclass
import numpy as np
from f1.pendulum_simulator import PendulumConfig, _rk4_step


@dataclass(frozen=True)
class FilterConfig:
    acceleration_noise: float = 0.004
    initial_angle_std: float = np.deg2rad(45)
    initial_velocity_std: float = 1.0


@dataclass(frozen=True)
class RamseyConfig:
    gradient_tesla_per_m: float = 100e-9
    gyromagnetic_ratio: float = 2 * np.pi * 28e9
    coherence_time: float = 150e-6
    readout_contrast: float = 0.95
    shots: int = 256
    parallel_spins: int = 64
    readout_time: float = 5e-6
    fixed_tau: float = 25e-6
    candidate_taus: tuple = (25e-6, 50e-6, 100e-6, 150e-6, 200e-6)


def transition(state, dt, physics=PendulumConfig()):
    return _rk4_step(0.0, np.asarray(state, float), dt, physics, dt)


def transition_jacobian(state, dt, physics=PendulumConfig()):
    # Central differences of the actual RK4 transition (not an Euler Jacobian).
    result = np.empty((2, 2))
    for j in range(2):
        offset = np.eye(2)[j] * 1e-5
        result[:, j] = (transition(state + offset, dt, physics) -
                        transition(state - offset, dt, physics)) / 2e-5
    return result


def predict(state, covariance, dt, physics, cfg):
    jac = transition_jacobian(state, dt, physics)
    process = cfg.acceleration_noise * np.array(
        [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    return transition(state, dt, physics), jac @ covariance @ jac.T + process


def update(state, covariance, measurement, expected, jacobian, variance):
    h = np.asarray(jacobian, float)
    variance = max(float(variance), 1e-12)
    innovation = float(measurement - expected)
    innovation_variance = float(h @ covariance @ h + variance)
    gain = covariance @ h / innovation_variance
    updated = state + gain * innovation
    residual = np.eye(2) - np.outer(gain, h)
    # Joseph form preserves positive semidefiniteness during repeated updates.
    cov = residual @ covariance @ residual.T + variance * np.outer(gain, gain)
    return updated, (cov + cov.T) / 2, innovation**2 / innovation_variance


def ramsey_response(theta, tau, reference=0., sensor=RamseyConfig(), length=1.):
    coupling = sensor.gyromagnetic_ratio * sensor.gradient_tesla_per_m * length * tau
    phase = coupling * np.sin(theta) - reference
    visibility = sensor.readout_contrast * np.exp(-tau / sensor.coherence_time)
    probability = (1 + visibility * np.sin(phase)) / 2
    derivative = visibility * np.cos(phase) * coupling * np.cos(theta) / 2
    return probability, derivative, visibility, coupling


def fisher_information(theta, tau, reference=0., sensor=RamseyConfig(), length=1.):
    p, dp, visibility, coupling = ramsey_response(theta, tau, reference, sensor, length)
    classical_fi = sensor.shots * dp**2 / max(float(p * (1-p)), 1e-12)
    # QFI belongs to the spin state before detector contrast losses.
    coherence = np.exp(-tau / sensor.coherence_time)
    quantum_fi = sensor.shots * (coherence * coupling * np.cos(theta))**2
    return float(classical_fi), float(quantum_fi)


def estimate(dataset, noise_std, mode="classical", physics=PendulumConfig(),
             cfg=FilterConfig(), sensor=RamseyConfig(), seed=0, policy="always"):
    """Estimate from classical observations or newly simulated Ramsey outcomes.

    Ramsey frequencies use a Gaussian EKF approximation to binomial readout.
    This approximation and coherence parameters require physical validation.
    Event policy uses uncertainty and innovation only to request readings.
    """
    if mode not in ("classical", "ramsey_fixed", "ramsey_adaptive"):
        raise ValueError("unknown estimator mode")
    time = dataset["time"]
    dt = float(time[1] - time[0])
    rng = np.random.default_rng(seed)
    state = np.zeros(2)
    cov = np.diag([cfg.initial_angle_std**2, cfg.initial_velocity_std**2])
    states, covs = np.empty((len(time), 2)), np.empty((len(time), 2, 2))
    nis, taus, info, qfi = [np.full(len(time), np.nan) for _ in range(4)]
    used = np.zeros(len(time), bool)
    frequencies = np.full(len(time), np.nan)
    last_reading, distrust_until = -1e6, -1e6
    for k, t in enumerate(time):
        if k:
            state, cov = predict(state, cov, dt, physics, cfg)
        available = bool(dataset["observed_mask"][k])
        request = (policy == "always" or np.sqrt(cov[0, 0]) > np.deg2rad(.6)
                   or t - last_reading >= .2 or t <= distrust_until)
        if available and request:
            if mode == "classical":
                state, cov, nis[k] = update(state, cov, dataset["observations"][k],
                                            state[0], [1., 0.], noise_std**2)
            else:
                tau, reference = sensor.fixed_tau, 0.
                if mode == "ramsey_adaptive":
                    candidates = []
                    for candidate in sensor.candidate_taus:
                        _, _, _, coupling = ramsey_response(state[0], candidate, sensor=sensor,
                                                            length=physics.length)
                        # Limit phase ambiguity under the uncertain prior.
                        if abs(coupling) * np.sqrt(cov[0, 0]) <= .65:
                            phase_ref = coupling * np.sin(state[0])
                            fi, _ = fisher_information(state[0], candidate, phase_ref,
                                                       sensor, physics.length)
                            candidates.append((fi, candidate, phase_ref))
                    if candidates:
                        _, tau, reference = max(candidates)
                batch_time = np.ceil(sensor.shots / sensor.parallel_spins) * (
                    tau + sensor.readout_time)
                if batch_time > dt:
                    raise ValueError("Ramsey shot batch exceeds the simulation sampling interval")
                actual_theta = dataset["true_state"][k, 0] + dataset["sensor_bias"][k]
                p_true, _, _, _ = ramsey_response(actual_theta, tau, reference,
                                                 sensor, physics.length)
                frequency = rng.binomial(sensor.shots, float(p_true)) / sensor.shots
                p, dp, _, _ = ramsey_response(state[0], tau, reference, sensor, physics.length)
                info[k], qfi[k] = fisher_information(state[0], tau, reference,
                                                    sensor, physics.length)
                state, cov, nis[k] = update(state, cov, frequency, p, [dp, 0.],
                                            float(p*(1-p)) / sensor.shots)
                taus[k], frequencies[k] = tau, frequency
            used[k] = True
            last_reading = t
            if nis[k] > 6.635:  # chi-square(1) 99% innovation threshold
                distrust_until = t + .2
        states[k], covs[k] = state, cov
    return {"state": states, "covariance": covs, "nis": nis, "used": used,
            "tau": taus, "fisher_information": info, "qfi": qfi,
            "frequencies": frequencies}


def estimation_metrics(truth, result):
    error = result["state"] - truth
    std = np.sqrt(np.diagonal(result["covariance"], axis1=1, axis2=2))
    coverage = np.mean(abs(error) <= 1.96 * std, axis=0)
    nees = np.einsum("ni,nij,nj->n", error, np.linalg.inv(result["covariance"]), error)
    return {"angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error[:, 0]**2)))),
            "velocity_rmse_rad_s": float(np.sqrt(np.mean(error[:, 1]**2))),
            "angle_95_coverage": float(coverage[0]), "velocity_95_coverage": float(coverage[1]),
            "mean_nees": float(np.mean(nees)), "mean_nis": float(np.nanmean(result["nis"])),
            "measurements": int(np.sum(result["used"])),
            "untrusted_fraction": float(np.mean((std[:, 0] > np.deg2rad(2)) |
                                                  (np.nan_to_num(result["nis"]) > 6.635)))}
