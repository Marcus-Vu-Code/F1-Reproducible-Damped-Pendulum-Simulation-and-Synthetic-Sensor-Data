"""Classical and exactly simulated two-qubit kernel surrogates."""
from dataclasses import dataclass
import numpy as np


def polynomial_features(x):
    theta, omega = x.T
    return np.column_stack([np.ones(len(x)), theta, omega, np.sin(theta),
                            np.cos(theta), theta*omega, theta**2, omega**2,
                            theta**3, theta**2*omega, theta*omega**2, omega**3])


def feature_states(x):
    """RY encoding, CNOT, local RZ re-encoding, and a second RY layer.

    Two qubits, six single-qubit gates and two CNOT gates. The CNOT entangles
    input-dependent states. This tiny feature map is efficiently classical.
    """
    state = np.zeros((len(x), 4), complex)
    state[:, 0] = 1
    for layer in range(2):
        for qubit in range(2):
            angle = x[:, qubit] / (layer + 1)
            c, s = np.cos(angle/2), np.sin(angle/2)
            pairs = [(0, 2), (1, 3)] if qubit == 0 else [(0, 1), (2, 3)]
            for a, b in pairs:
                first, second = state[:, a].copy(), state[:, b].copy()
                state[:, a] = c*first - s*second
                state[:, b] = s*first + c*second
        state[:, [2, 3]] = state[:, [3, 2]]  # CNOT: q0 controls q1
        if layer == 0:
            state *= np.exp(-.5j * x[:, 0, None] * np.array([1, 1, -1, -1]))
            state *= np.exp(-.5j * x[:, 1, None] * np.array([1, -1, 1, -1]))
    return state


def fidelity_kernel(x, z):
    return np.abs(feature_states(x).conj() @ feature_states(z).T)**2


@dataclass
class Surrogate:
    kind: str
    regularization: float = 1e-4
    length_scale: float = 1.0

    def fit(self, x, target):
        self.center, self.scale = x.mean(axis=0), x.std(axis=0) + 1e-12
        self.x_train = (x - self.center) / self.scale
        self.target_center = target.mean(axis=0)
        centered = target - self.target_center
        if self.kind == "polynomial":
            features = polynomial_features(x)
            self.weights = np.linalg.solve(features.T @ features + self.regularization *
                                            np.eye(features.shape[1]), features.T @ centered)
        else:
            kernel = self.kernel(self.x_train, self.x_train)
            self.weights = np.linalg.solve(kernel + self.regularization * np.eye(len(x)), centered)
        self.residual_covariance = np.eye(2)*1e-8
        return self

    def calibrate(self, x, target):
        residual = self.predict(x)-x-target
        self.residual_covariance = np.cov(residual.T) + np.eye(2)*1e-8
        return self

    def predictive_covariance(self, x, covariance):
        x = np.atleast_2d(x)
        jacobian = np.empty((len(x), 2, 2))
        for j in range(2):
            offset = np.eye(2)[j]*1e-5
            jacobian[:, :, j] = (self.predict(x+offset)-self.predict(x-offset))/2e-5
        return (jacobian @ covariance @ jacobian.transpose(0, 2, 1) +
                self.residual_covariance)

    def kernel(self, x, z):
        if self.kind == "quantum":
            return fidelity_kernel(x / self.length_scale, z / self.length_scale)
        distance = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(z*z, axis=1)[None, :]
                              - 2*x @ z.T, 0)
        return np.exp(-distance / (2*self.length_scale**2))

    def predict(self, x, depolarization=0.):
        x = np.atleast_2d(x)
        if self.kind == "polynomial":
            features = polynomial_features(x)
        else:
            features = self.kernel((x-self.center)/self.scale, self.x_train)
            if self.kind == "quantum" and depolarization:
                # Test-only mixed query rho=(1-d)|psi><psi|+d*I/4.
                features = (1-depolarization)*features + depolarization/4
        return x + self.target_center + features @ self.weights

    def save(self, path):
        np.savez_compressed(path, kind=self.kind, regularization=self.regularization,
                            length_scale=self.length_scale, center=self.center, scale=self.scale,
                            x_train=self.x_train, target_center=self.target_center, weights=self.weights,
                            residual_covariance=self.residual_covariance)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(str(data["kind"]), float(data["regularization"]), float(data["length_scale"]))
            for name in ("center", "scale", "x_train", "target_center", "weights", "residual_covariance"):
                setattr(model, name, data[name])
        return model


def select_model(x, target, val_x, val_target, quantum=False):
    scores, models = [], []
    for kind in (["quantum"] if quantum else ["polynomial", "rbf"]):
        for regularization in (1e-5, 1e-3, 1e-1):
            for length_scale in ((.5, 1., 2.) if kind != "polynomial" else (1.,)):
                model = Surrogate(kind, regularization, length_scale).fit(x, target)
                mse = float(np.mean(((model.predict(val_x) - val_x - val_target) /
                                     np.array([1., 3.]))**2))
                models.append(model)
                scores.append({"kind": kind, "regularization": regularization,
                               "length_scale": length_scale, "validation_mse": mse})
    best = int(np.argmin([item["validation_mse"] for item in scores]))
    return models[best].calibrate(val_x, val_target), scores
