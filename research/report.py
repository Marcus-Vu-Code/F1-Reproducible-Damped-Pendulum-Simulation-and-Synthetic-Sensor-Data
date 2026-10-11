"""Generate a progress report from experiment output, without hand-entered metrics."""
import argparse
import json
from pathlib import Path


def generate(root):
    root = Path(root)
    s = json.loads((root/"summary.json").read_text())
    lines = ["# Pendulum digital twin: research progress", "", "Experiment date: October 8, 2026.", "",
             "## Scope and status", "",
             "The repository now contains a working simulation pipeline from sensing to state estimation and prediction. "
             "F1 and F2 have implemented baselines. F3 provides an idealized quantum sensing hypothesis. "
             "F4 provides classical surrogate selection. F5 provides an exactly simulated quantum kernel. "
             "F6 integrates the four combinations and exports preliminary results. "
             "Implementation does not establish quantum advantage or complete every research milestone.", "",
             "Spring prototypes include first-order prediction uncertainty, filter-guided measurement requests, "
             "innovation-based distrust flags, and OOD/noise stress tests. Autonomous actuator control, "
             "validated trust calibration, recovery guarantees and a publication-ready study remain open.", "",
             "## Experimental method", "",
             "The default run uses 12 independent initial conditions, shared across four scenarios: 6 training, "
             "2 validation and 4 test trajectories. Each run spans 10 seconds at 20 ms resolution. "
             "Initial angles lie within +/-50 degrees and velocities within +/-0.4 rad/s. "
             "Splits operate on whole trajectories. Models train on 256 pairs from nominal conditions only "
             "and predict 100 ms ahead. Hyperparameters and classical model family use validation estimated-state targets. "
             "Ground truth enters only final evaluation, not model fitting or sensor controls.", "",
             "Each model predicts a state increment from the current estimated state. Targets come from the same "
             "estimator stream at a future time. This introduces estimator bias and temporally correlated targets. "
             "Test RMSE is computed for each trajectory and then averaged. Standard deviations describe "
             "between-trajectory variation rather than confidence intervals. One-second forecasts recursively "
             "apply the model ten times without further measurements.", "",
             "## F2: classical estimation", "",
             "The EKF predicts with the nonlinear RK4 transition, linearizes that discrete transition, and updates "
             "with noisy angle readings. It uses the Joseph covariance update. White acceleration noise supplies "
             "the process covariance. Its nominal damping remains 0.08 even when truth drifts to 0.16.", "",
             "| Scenario | Angle EKF RMSE (deg) | Velocity EKF RMSE (rad/s) | Angle 95% coverage |", "|---|---:|---:|---:|"]
    for r in s["estimation"]:
        if r["estimator"] == "classical":
            lines.append(f"| {r['scenario']} | {r['angle_rmse_deg']:.3f} | {r['velocity_rmse_rad_s']:.4f} | {100*r['angle_95_coverage']:.1f}% |")
    lines += ["", "## F3: simulated Ramsey sensing", "",
              "Assume a pendulum moving in an engineered uniform transverse magnetic gradient. "
              "Its local field is B(theta)=G L sin(theta), and a spin sensor accumulates phase phi=gamma B tau. "
              "The binary-readout probability is p=[1+V sin(phi-reference)]/2, with V=contrast exp(-tau/T2). "
              "Readings are sampled from a binomial distribution. The EKF updates with the observed shot frequency "
              "using a Gaussian approximation. Gradient, contrast, coherence and parallelism are illustrative parameters.", "",
              "A fixed strategy uses 25 us and zero reference phase. The adaptive strategy aligns the "
              "readout with the predicted phase and selects tau from 25, 50, 100, 150 or 200 us. "
              "It restricts phase uncertainty to avoid local ambiguities. The control never observes truth. "
              "For independent spins, QFI=N C^2 (dphi/dtheta)^2, with C=exp(-tau/T2), "
              "and measurement FI=N (dp/dtheta)^2/[p(1-p)]. "
              "The code checks FI<=QFI. Detector contrast reduces achievable information below the bound.", "",
              "Nominal T2 is 150 us, gamma=2 pi 28 GHz/T, G=100 nT/m and contrast=0.95. "
              "Each batch uses 256 shots across 64 ideal parallel spins. High noise reduces T2 by three for "
              "Ramsey sensing, while the classical angle noise rises from 1 to 4 degrees. These are modality-specific "
              "stress settings, not matched physical noise sources. All shot batches must fit the 20 ms sampling interval. "
              "The field is assumed frozen during each batch. The model is not a validated NV sensor or a dipole magnet model.", "",
              "| Shots per batch | Fixed Ramsey RMSE (deg) | Adaptive Ramsey RMSE (deg) |", "|---:|---:|---:|"]
    for shots in (64,256,1024):
        fixed=next(r for r in s["shot_efficiency"] if r["shots"]==shots and r["estimator"]=="ramsey_fixed")
        adaptive=next(r for r in s["shot_efficiency"] if r["shots"]==shots and r["estimator"]=="ramsey_adaptive")
        lines.append(f"| {shots} | {fixed['angle_rmse_deg']:.3f} | {adaptive['angle_rmse_deg']:.3f} |")
    lines += ["", "Matching shot counts does not match total interrogation time. summary.json also records total "
              "shots and summed spin-interrogation seconds. The classical sensor has an externally assumed noise "
              "variance rather than a physical readout budget. No cross-modality sensing advantage follows from this experiment.", "",
              "## F4/F5: surrogate models", "",
              "Classical selection considers ridge regression on polynomial/trigonometric features and an RBF kernel. "
              "The quantum model uses a two-qubit RY/CNOT/RZ feature circuit, six single-qubit gates and two CNOTs. "
              "Its fidelity kernel is |<psi(x)|psi(z)>|^2. Classical ridge regression fits the resulting kernel. "
              "Both branches use the same training budget and trajectory splits. Input scaling uses training data only. "
              "Model selection uses regularization 1e-5, 1e-3, 1e-1 and kernel scales 0.5, 1, 2.", "",
              "The two-qubit state has four complex amplitudes and the fidelity kernel has finite feature rank at most 16. "
              "Exact NumPy evaluation is inexpensive. This implements a quantum feature-map prototype, not a speedup. "
              "The main kernel is noiseless and has no shot-estimation error. A separate inference-only ablation mixes "
              "query states with I/4, while retaining noiseless training states.", "", "Selected models:", ""]
    for key,value in s["metadata"]["selections"].items():
        lines.append(f"- {key}: {value['selected']}")
    lines += ["", "## F6: four-way comparison", "",
              "Values are mean 100 ms angle forecast RMSE in degrees across the four held-out trajectories. "
              "The classical family may differ between sensor streams because each uses validation selection.", "",
              "| Condition | Angle + classical | Angle + quantum | Ramsey + classical | Ramsey + quantum |", "|---|---:|---:|---:|---:|"]
    for scenario in ("nominal","high_noise","sparse_measurements","parameter_and_sensor_drift"):
        row=[scenario]
        for mode,model in [("classical","classical"),("classical","quantum"),("ramsey_adaptive","classical"),("ramsey_adaptive","quantum")]:
            r=next(r for r in s["prediction"] if r["scenario"]==scenario and r["estimator"]==mode and r["model"]==model)
            row.append(f"{r['angle_rmse_deg']:.3f}")
        lines.append("| "+" | ".join(row)+" |")
    lines += ["", "The tested classical combination gives the smallest mean angle forecast error in all four "
              "conditions. Adaptive sensing improves the Ramsey baseline. The quantum kernel has larger long-horizon "
              "and out-of-distribution errors. These conclusions apply only to the chosen simulator, feature map, "
              "hyperparameter search and small sample. They do not rule out different quantum methods.", "",
              "## Uncertainty and Spring prototypes", "",
              "Estimator covariance gives marginal 95% intervals. NIS and NEES diagnose model consistency with "
              "reference means of 1 and 2. Prediction covariance uses J P J^T plus validation residual covariance. "
              "This first-order approximation assumes independence and does not account fully for estimator-target "
              "correlation or distribution shift. Empirical prediction coverage appears in summary.json.", ""]
    always=next(r for r in s["measurement_policy"] if r["scenario"]=="nominal" and r["policy"]=="always")
    event=next(r for r in s["measurement_policy"] if r["scenario"]=="nominal" and r["policy"]=="event")
    lines += [f"The covariance/innovation measurement policy reduces nominal readings from {always['measurements']:.0f} "
              f"to {event['measurements']:.2f} per run ({100*(1-event['measurements']/always['measurements']):.1f}% fewer). "
              f"Angle RMSE increases from {always['angle_rmse_deg']:.3f} to {event['angle_rmse_deg']:.3f} degrees. "
              "It requests a measurement above 0.6-degree predicted uncertainty, after a 0.2-second gap, "
              "or for 0.2 seconds after an innovation exceeds the 99% chi-square threshold. "
              "This filter-guided heuristic constitutes a partial sequential measurement policy; it does not "
              "implement an RL/Bayesian control agent or certify safety.", "",
              "Stress tests evaluate 75–84-degree initial angles outside the training range, classical sensor noise, "
              "Ramsey coherence loss, sparse data, damping/bias drift, and inference-time quantum-state mixing. "
              "Unfamiliar states cause large quantum-kernel rollout errors. Recovery and recalibration mechanisms "
              "still require evaluation.", "", "## Limitations and next experiments", ""]
    lines += ["- "+x for x in s["metadata"]["limitations"]]
    lines += ["- Repeat the ensemble across multiple seeds and substantially more held-out trajectories.",
              "- Validate sensing geometry with Dr. Kumar and match sensing time or energy across modalities.",
              "- Add bias/damping augmentation and measure recovery after abrupt changes.",
              "- Evaluate finite-shot kernel estimation and physically motivated circuit noise.",
              "- Calibrate trust thresholds on validation failures before downstream control.", "",
              "## Reproduction", "", "```bash", "python -m pip install -r requirements.txt",
              "python -m research.run", "python -m unittest discover -s tests -v",
              "python -m pip install -r requirements-plots.txt", "python -m research.figures", "```", "",
              "Use Python 3.11+ with the pinned NumPy version. Seeds and versions target consistent numerical results. "
              "Small platform-dependent floating-point differences or compressed archive byte differences may occur. "
              "Source code and output manifests contain no machine-specific runtime paths.", "",
              "## References", "",
              "- Project Description for Marcus Vu.pdf, supplied project brief, Fall and Spring milestones.",
              "- Degen, Reinhard and Cappellaro (2017), [Quantum sensing](https://arxiv.org/abs/1611.02427). "
              "Background for the spin-sensing framework.",
              "- Havlicek et al. (2019), [Supervised learning with quantum-enhanced feature spaces](https://arxiv.org/abs/1804.11326). "
              "Background for quantum feature maps and kernels. Our circuit is an original small prototype.", ""]
    destination=root/"progress_report.md"
    destination.write_text("\n".join(lines),encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("results/progress"))
    generate(parser.parse_args().results)
