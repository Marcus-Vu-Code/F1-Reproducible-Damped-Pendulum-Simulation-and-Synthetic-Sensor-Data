import tempfile
import unittest
from pathlib import Path
import numpy as np
from f1.pendulum_simulator import PendulumConfig, Scenario, generate_dataset, simulate_truth
from research.estimation import (RamseyConfig, estimate, estimation_metrics, fisher_information,
                                 ramsey_response, transition, transition_jacobian)
from research.models import Surrogate, feature_states, fidelity_kernel
from research.run import split_for


class ResearchTests(unittest.TestCase):
    def test_classical_ekf_improves_noisy_angle(self):
        scenario = Scenario("test", "test", duration=3., dt=.02,
                            measurement_noise_std=np.deg2rad(4.), seed=11)
        data = generate_dataset(scenario)
        result = estimate(data, scenario.measurement_noise_std)
        sensor_rmse = np.sqrt(np.mean((data["observations"]-data["true_state"][:, 0])**2))
        ekf_rmse = np.sqrt(np.mean((result["state"][:, 0]-data["true_state"][:, 0])**2))
        self.assertLess(ekf_rmse, sensor_rmse/2)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(result["covariance"])), 0.)

    def test_rk4_matches_fine_integrator(self):
        cfg = PendulumConfig()
        _, fine, _ = simulate_truth(cfg, 1., .001)
        _, coarse, _ = simulate_truth(cfg, 1., .02)
        np.testing.assert_allclose(coarse, fine[::20], atol=2e-6)

    def test_transition_jacobian_matches_linear_limit(self):
        dt = 1e-4
        jac = transition_jacobian(np.zeros(2), dt)
        continuous = np.array([[0., 1.], [-9.81, -.08]])
        np.testing.assert_allclose(jac, np.eye(2)+dt*continuous, atol=1e-7)

    def test_ramsey_derivative_and_information_bound(self):
        for theta in (-.8, 0., .5):
            sensor = RamseyConfig()
            p, derivative, _, _ = ramsey_response(theta, 100e-6, .1, sensor)
            offset = 1e-6
            finite = (ramsey_response(theta+offset, 100e-6, .1, sensor)[0]-
                      ramsey_response(theta-offset, 100e-6, .1, sensor)[0])/(2*offset)
            self.assertAlmostEqual(float(derivative), float(finite), places=8)
            self.assertTrue(0 < p < 1)
            fi, qfi = fisher_information(theta, 100e-6, .1, sensor)
            self.assertLessEqual(fi, qfi+1e-9)
        coupling = sensor.gyromagnetic_ratio * sensor.gradient_tesla_per_m * 100e-6
        fi, qfi = fisher_information(.4, 100e-6, coupling*np.sin(.4), sensor)
        self.assertAlmostEqual(fi, sensor.readout_contrast**2 * qfi, places=8)

    def test_ramsey_is_reproducible_and_respects_shot_schedule(self):
        data = generate_dataset(Scenario("test", "test", duration=1., dt=.02, seed=5))
        first = estimate(data, .02, "ramsey_adaptive", seed=50)
        second = estimate(data, .02, "ramsey_adaptive", seed=50)
        np.testing.assert_array_equal(first["state"], second["state"])
        self.assertTrue(np.all(first["fisher_information"] <= first["qfi"]+1e-8))
        self.assertTrue(np.all(np.linalg.eigvalsh(first["covariance"])>0))

    def test_quantum_states_normalized_kernel_psd(self):
        x = np.random.default_rng(5).normal(size=(20, 2))
        state = feature_states(x)
        np.testing.assert_allclose(np.sum(abs(state)**2, axis=1), 1., atol=1e-12)
        kernel = fidelity_kernel(x, x)
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(kernel), 1., atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(kernel).min(), -1e-10)
        # A product-state 2x2 amplitude matrix has zero determinant.
        self.assertGreater(abs(np.linalg.det(state[0].reshape(2, 2))), .01)

    def test_model_round_trip_and_training_fit(self):
        x = np.random.default_rng(2).normal(size=(40, 2))
        delta = np.column_stack([.1*x[:, 1], -.1*np.sin(x[:, 0])])
        for kind in ("polynomial", "rbf", "quantum"):
            model = Surrogate(kind, 1e-6).fit(x, delta)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"model.npz"
                model.save(path)
                loaded = Surrogate.load(path)
                np.testing.assert_allclose(model.predict(x), loaded.predict(x), atol=1e-12)
            # The small quantum feature map need not exactly represent this function.
            # Require improvement over persistence, which predicts a zero increment.
            self.assertLess(np.mean((model.predict(x)-x-delta)**2), np.mean(delta**2)/2)
            covariance = model.predictive_covariance(x, np.tile(np.eye(2)*.01, (len(x), 1, 1)))
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance)>0))

    def test_whole_trajectory_split(self):
        ids = {split: {i for i in range(12) if split_for(i, 6, 2)==split}
               for split in ("train", "validation", "test")}
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["validation"] & ids["test"])
        self.assertEqual([len(ids[s]) for s in ("train", "validation", "test")], [6, 2, 4])


if __name__ == "__main__":
    unittest.main()
