"""Export scientific figures: python -m research.figures --results results/progress."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#167D8D", "#D78932", "#7562A6", "#BE5465"]
LABELS = {"classical": "Classical EKF", "ramsey_fixed": "Fixed Ramsey",
          "ramsey_adaptive": "Adaptive Ramsey"}
SHORT = ["Nominal", "High noise", "Sparse", "Drift"]


def generate(results):
    root = Path(results)
    summary = json.loads((root/"summary.json").read_text())
    folder = root/"figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    def save(fig, name):
        for extension in ("png", "svg"):
            fig.savefig(folder/f"{name}.{extension}")
        plt.close(fig)
    manifest = json.loads((root/"manifest.json").read_text())
    sample = next(r for r in manifest if r["split"]=="test" and r["scenario"]=="nominal")
    data = np.load(root/"trajectories"/sample["file"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(data["time"], np.rad2deg(data["true_state"][:, 0]), color="#253445", lw=2, label="Truth")
    axes[0].scatter(data["time"][::5], np.rad2deg(data["observations"][::5]),
                    color="#B6BEC5", s=9, label="Angle readings")
    for mode, color in zip(("classical", "ramsey_adaptive"), COLORS):
        axes[0].plot(data["time"], np.rad2deg(data[f"{mode}_state"][:, 0]), color=color, lw=1, label=LABELS[mode])
        axes[1].plot(data["time"], data[f"{mode}_state"][:, 1], color=color, label=LABELS[mode])
    axes[1].plot(data["time"], data["true_state"][:, 1], color="#253445", lw=2, label="Truth")
    axes[0].set_ylabel("Angle (degrees)"); axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].set_xlabel("Time (s)"); axes[0].legend(ncols=2, fontsize=10)
    fig.tight_layout(); save(fig, "state_estimation")
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (mode, color) in enumerate(zip(LABELS, COLORS)):
        rows = [r for r in summary["estimation"] if r["estimator"]==mode]
        ax.bar(np.arange(4)+(j-1)*.24, [r["angle_rmse_deg"] for r in rows], .24,
               yerr=[r["angle_rmse_deg_std"] for r in rows], color=color, label=LABELS[mode], capsize=3)
    ax.set_xticks(range(4), SHORT); ax.set_ylabel("Angle RMSE (degrees)")
    ax.legend(); fig.tight_layout(); save(fig, "estimation_rmse")
    fig, ax = plt.subplots(figsize=(10, 5))
    configs = [("classical", "classical"), ("classical", "quantum"),
               ("ramsey_adaptive", "classical"), ("ramsey_adaptive", "quantum")]
    for j, ((mode, model), color) in enumerate(zip(configs, COLORS)):
        rows = [r for r in summary["prediction"] if r["estimator"]==mode and r["model"]==model]
        label = ("Angle sensor" if mode=="classical" else "Ramsey sensor")+" + "+("classical model" if model=="classical" else "quantum kernel")
        ax.bar(np.arange(4)+(j-1.5)*.19, [r["angle_rmse_deg"] for r in rows], .19,
               yerr=[r["angle_rmse_deg_std"] for r in rows], color=color, label=label, capsize=2)
    ax.set_xticks(range(4), SHORT); ax.set_ylabel("100 ms forecast angle RMSE (degrees)")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "four_way_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode, color in zip(("ramsey_fixed", "ramsey_adaptive"), COLORS):
        rows = [r for r in summary["shot_efficiency"] if r["estimator"]==mode]
        axes[0].plot([r["shots"] for r in rows], [r["angle_rmse_deg"] for r in rows], "o-", color=color, label=LABELS[mode])
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("Shots per measurement batch")
    axes[0].set_ylabel("Angle RMSE (degrees)"); axes[0].legend()
    for (mode, model), color in zip(configs, COLORS):
        rows = [r for r in summary["learning_curves"] if r["estimator"]==mode and r["model"]==model]
        axes[1].plot([r["samples"] for r in rows], [r["angle_rmse_deg"] for r in rows], "o-", color=color,
                     label=f"{'Angle' if mode=='classical' else 'Ramsey'} + {model}")
    axes[1].set_xlabel("Training pairs"); axes[1].set_ylabel("Forecast RMSE (degrees)")
    axes[1].legend(fontsize=9); fig.tight_layout(); save(fig, "efficiency")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for mode, color in zip(("classical", "ramsey_adaptive"), COLORS):
        rows = [r for r in summary["estimation"] if r["estimator"]==mode]
        ax.plot(SHORT, [100*r["angle_95_coverage"] for r in rows], "o-", color=color, label=LABELS[mode])
    ax.axhline(95, color="#777777", linestyle="--", label="Nominal 95%")
    ax.set_ylabel("Empirical angle interval coverage (%)"); ax.set_ylim(85, 101)
    ax.legend(); fig.tight_layout(); save(fig, "uncertainty_coverage")
    # Editable slide data shares the same underlying experiment summary.
    example = {"time": data["time"][::10].tolist(),
               "truth_angle_deg": np.rad2deg(data["true_state"][::10, 0]).tolist(),
               "ekf_angle_deg": np.rad2deg(data["classical_state"][::10, 0]).tolist(),
               "ramsey_angle_deg": np.rad2deg(data["ramsey_adaptive_state"][::10, 0]).tolist()}
    (root/"slide_data.json").write_text(json.dumps({"summary": summary, "example": example}, indent=2), encoding="utf-8")
    print(f"Figures written to {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/progress"))
    generate(parser.parse_args().results)
