# Pendulum simulation and digital-twin research prototype

This repository contains a reproducible damped-pendulum simulator and an
experimental sensing, estimation and prediction pipeline for the research
project under Dr. Kumar. All experiments run in software.

The original F1 datasets remain in `data/f1_pendulum`. The `research` package
adds classical estimation, idealized Ramsey spin sensing, classical surrogate
models, and an exactly simulated two-qubit kernel. These support the Fall
project's four-way comparison. Quantum sensing parameters are illustrative,
and the small quantum model establishes no computational or sensing advantage.

## Current progress

| Milestone | Implementation | Remaining research |
| --- | --- | --- |
| F1 | Nonlinear simulator, noise/sparsity/drift datasets, whole-trajectory splits | Larger ensembles |
| F2 | EKF, RMSE, covariance, NIS/NEES diagnostics | Bias/parameter adaptation |
| F3 | Binomial Ramsey measurement model, adaptive phase/time, FI/QFI, shot sweep | Physical validation and matched resource budgets |
| F4 | Polynomial/RBF surrogate selection and learning curves | Broader classical baselines |
| F5 | Two-qubit fidelity kernel, checkpoints, query-noise ablation | Finite-shot kernels and physical circuit noise |
| F6 | Four sensing/model combinations, short forecasts and rollouts | Larger statistical study |

Spring prototypes add approximate prediction uncertainty, sequential reading
requests, diagnostic distrust flags, and OOD/noise stress tests. Autonomous
actuator control and validated safety/recovery policies remain open.

The default nominal result is 0.31-degree mean 100 ms angle forecast RMSE for
classical sensing + classical modeling, versus 0.88 for the same sensing +
quantum kernel. Results average four held-out test trajectories and remain
preliminary. Read the generated [progress report](results/progress/progress_report.md)
for all assumptions and comparisons.

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

The generated datasets use explicit random seeds. Fixed parameters and the
pinned NumPy version target consistent numerical results across machines.
Small platform-dependent floating-point differences may occur. Compressed
archive byte identity is not the reproducibility criterion.

## Run the complete experiment

After setup, substitute your virtual environment's Python executable for
`python` in these platform-neutral commands:

```bash
python -m research.run
python -m research.report
```

The default run generates 12 independent initial conditions across four
scenarios: 6 train, 2 validation, 4 test. Each simulation lasts 10 seconds with
20 ms integration. The model predicts 100 ms ahead and repeats ten predictions
for a 1-second rollout. Training and validation use future estimated states as
targets. Ground truth enters evaluation, not model fitting or sensing controls.
Models train on nominal conditions only, with 256 training pairs.

Outputs under `results/progress`:

- `summary.json`: averaged metrics, parameters, selected models and limitations
- `per_trajectory.json`: individual metrics underlying every comparison
- `manifest.json`: complete trajectory IDs and train/validation/test membership
- `models/*.npz`: loadable surrogate checkpoints
- `trajectories/*.npz`: truth, sensor data, estimated states and covariances
- `progress_report.md`: generated research progress report

Trajectory output is ignored by Git because anyone can regenerate it. The
original four F1 datasets remain tracked.

For a quicker smoke run:

```bash
python -m research.run --output results/smoke --train 2 --validation 1 --test 1 --budget 64
```

To use a saved surrogate:

```python
import numpy as np
from research.models import Surrogate

model = Surrogate.load("results/progress/models/classical_classical.npz")
next_state = model.predict(np.array([[0.2, 0.0]]))  # radians, radians/second
```

## Export figures

```bash
python -m pip install -r requirements-plots.txt
python -m research.figures
```

This exports scientific plots in PNG and SVG plus editable source data for the
progress slides. The core experiment requires only NumPy. The slide deck is in
`output/presentations/Pendulum_Research_Progress_2026-10-08_v2.pptx`, with speaker notes.

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

GitHub Actions runs the tests and a small integrated benchmark on Windows and
Linux with Python 3.11 and 3.12. Local tests check the integrator, EKF covariance,
Ramsey derivatives and information bound, kernel normalization/PSD,
checkpoint round trips, and whole-trajectory splitting.

## Research references

- [Degen et al. (2017), Quantum sensing](https://arxiv.org/abs/1611.02427)
- [Havlicek et al. (2019), Supervised learning with quantum-enhanced feature spaces](https://arxiv.org/abs/1804.11326)

These motivate the sensing and kernel frameworks. The illustrative field
geometry and small feature circuit are implementation choices for this
prototype, not validated reproductions of a physical device or either paper.
