import tempfile
import unittest
from pathlib import Path

import numpy as np

from f1.pendulum_simulator import (
    PendulumConfig,
    Scenario,
    generate_dataset,
    simulate_truth,
    write_benchmark,
)


class F1SimulatorTests(unittest.TestCase):
    def test_truth_has_expected_shape_and_damped_energy(self):
        time, states, damping = simulate_truth(PendulumConfig(), duration=2.0, dt=0.01)
        self.assertEqual(time.shape, (201,))
        self.assertEqual(states.shape, (201, 2))
        self.assertEqual(damping.shape, (201,))
        self.assertTrue(np.all(np.isfinite(states)))
        energy = 0.5 * states[:, 1] ** 2 + 9.81 * (1 - np.cos(states[:, 0]))
        self.assertLess(energy[-1], energy[0])

    def test_sparse_measurements_are_aligned_to_grid(self):
        scenario = Scenario(
            name="test_sparse",
            description="test",
            duration=1.0,
            dt=0.01,
            measurement_stride=10,
            seed=1,
        )
        data = generate_dataset(scenario)
        self.assertEqual(int(np.sum(data["observed_mask"])), 11)
        self.assertTrue(np.all(np.isnan(data["observations"][~data["observed_mask"]])))
        self.assertTrue(np.all(np.isfinite(data["measurement_values"])))

    def test_drift_changes_damping_and_sensor_bias(self):
        scenario = Scenario(
            name="test_drift",
            description="test",
            duration=1.0,
            dt=0.01,
            damping_end=0.16,
            sensor_bias_rate=np.deg2rad(0.1),
            seed=2,
        )
        data = generate_dataset(scenario)
        self.assertAlmostEqual(float(data["damping"][0]), 0.08, places=8)
        self.assertAlmostEqual(float(data["damping"][-1]), 0.16, places=8)
        self.assertAlmostEqual(float(data["sensor_bias"][-1]), np.deg2rad(0.1), places=8)

    def test_benchmark_writes_all_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_benchmark(Path(directory))
            self.assertTrue(manifest.exists())
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 4)

    def test_generation_is_reproducible(self):
        scenario = Scenario(
            name="test_reproducible",
            description="test",
            duration=1.0,
            dt=0.01,
            measurement_stride=3,
            measurement_noise_std=0.2,
            seed=9876,
        )
        first = generate_dataset(scenario)
        second = generate_dataset(scenario)
        for key, value in first.items():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.array_equal(value, second[key], equal_nan=True), key)
            else:
                self.assertEqual(value, second[key], key)


if __name__ == "__main__":
    unittest.main()
