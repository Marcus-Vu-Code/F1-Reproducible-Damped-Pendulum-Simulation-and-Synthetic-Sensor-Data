"""F1 simulated physical environment."""

from .pendulum_simulator import (
    SCENARIOS,
    PendulumConfig,
    Scenario,
    generate_dataset,
    simulate_truth,
    write_benchmark,
)

__all__ = [
    "SCENARIOS",
    "PendulumConfig",
    "Scenario",
    "generate_dataset",
    "simulate_truth",
    "write_benchmark",
]
