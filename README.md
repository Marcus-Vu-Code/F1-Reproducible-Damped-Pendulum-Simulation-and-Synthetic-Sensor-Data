# Quantum-Enhanced Digital Twin — F1 simulated environment

This repository currently implements F1: a reproducible simulated physical
environment for later state-estimation and quantum-enhancement experiments.
It is software-only; no physical hardware is required for this milestone.

## Environment

The benchmark system is a damped nonlinear pendulum with latent state

```text
x(t) = [theta(t), omega(t)]

theta_dot = omega
omega_dot = -(g / L) sin(theta) - c(t) omega
```

The simulated sensor measures angle only:

```text
y_k = theta_k + b_k + v_k
```

where `v_k` is zero-mean Gaussian sensor noise and `b_k` is an optional
slowly changing sensor bias. The physical truth is integrated with fixed-step
fourth-order Runge–Kutta. The sensor observations are kept separate from the
truth so F2 can consume exactly the same datasets.

## F1 scenarios

| Dataset | Purpose |
| --- | --- |
| `nominal` | Regular measurements, 1 degree standard deviation noise |
| `high_noise` | Regular measurements, 4 degree standard deviation noise |
| `sparse_measurements` | A reading every 10 integration steps |
| `parameter_and_sensor_drift` | Damping increases from 0.08 to 0.16 and sensor bias ramps at 0.04 degrees/s |

Each dataset contains `time`, `true_state`, `damping`, `sensor_bias`, an
aligned `observations` array with `NaN` at unobserved times, `observed_mask`,
and compact measurement-only arrays. `manifest.json` documents all parameters,
seeds, and file names.

## Run locally or from GitHub

Requirements:

- Python 3.11 or newer
- The exact dependency listed in `requirements.txt`

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe f1\pendulum_simulator.py --output data\f1_pendulum
```

### macOS/Linux

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python f1/pendulum_simulator.py --output data/f1_pendulum
```

The generated datasets are deterministic because every scenario has an
explicit random seed. The pinned NumPy version and fixed simulation
parameters ensure that other users reproduce the same numerical results.

## Verify

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS/Linux:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

The tests also generate the same scenario twice and compare every output
array, so reproducibility is checked automatically.

F2 can now add the Extended Kalman Filter without changing the simulator.
The next research decision is intentionally deferred: whether F3 should add a
physically motivated quantum-sensing model, such as a simulated magnetometer,
or use a different quantum measurement abstraction.
