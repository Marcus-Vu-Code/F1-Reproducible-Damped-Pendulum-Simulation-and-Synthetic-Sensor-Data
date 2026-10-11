"""Generate the end-to-end benchmark: python -m research.run."""
import argparse
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from f1.pendulum_simulator import SCENARIOS, PendulumConfig, generate_dataset
from research.estimation import (FilterConfig, RamseyConfig, estimate, estimation_metrics,
                                 fisher_information, transition)
from research.models import select_model


MODES = ("classical", "ramsey_fixed", "ramsey_adaptive")


def split_for(index, train, validation):
    return "train" if index < train else "validation" if index < train+validation else "test"


def supervised_pairs(records, mode, split, horizon, budget=None):
    pairs = []
    for item in records:
        if item["split"] == split and item["scenario"] == "nominal":
            x = item["estimates"][mode]["state"]
            indices = np.arange(10, len(x)-horizon, 3)
            pairs.append((x[indices], x[indices+horizon]-x[indices]))
    x, y = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
    if budget is not None:
        # Deterministic sampling across whole training trajectories.
        indices = np.linspace(0, len(x)-1, min(budget, len(x)), dtype=int)
        x, y = x[indices], y[indices]
    return x, y


def prediction_metrics(truth, estimated, model, horizon, physics, dt, depolarization=0., covariance=None):
    start = np.arange(10, len(truth)-horizon, 5)
    predictions = model.predict(estimated[start], depolarization=depolarization)
    reference = truth[start+horizon]
    errors = predictions-reference
    persistence = estimated[start]-reference
    phys = estimated[start].copy()
    for _ in range(horizon):
        phys = np.array([transition(x, dt, physics) for x in phys])
    rollout_start = np.arange(10, len(truth)-10*horizon, 25)
    rollout = estimated[rollout_start].copy()
    for _ in range(10):
        rollout = model.predict(rollout, depolarization=depolarization)
    rollout_error = rollout-truth[rollout_start+10*horizon]
    metrics = {"angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(errors[:, 0]**2)))),
            "velocity_rmse_rad_s": float(np.sqrt(np.mean(errors[:, 1]**2))),
            "rollout_1s_angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(rollout_error[:, 0]**2)))),
            "rollout_1s_velocity_rmse_rad_s": float(np.sqrt(np.mean(rollout_error[:, 1]**2))),
            "persistence_angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(persistence[:, 0]**2)))),
            "physics_angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean((phys-reference)[:, 0]**2))))}
    if covariance is not None:
        forecast_covariance = model.predictive_covariance(estimated[start], covariance[start])
        forecast_std = np.sqrt(np.diagonal(forecast_covariance, axis1=1, axis2=2))
        metrics["prediction_angle_95_coverage"] = float(np.mean(abs(errors[:, 0]) <= 1.96*forecast_std[:, 0]))
    return metrics


def aggregate(rows, group_keys, metric_keys):
    result = []
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    for key, values in groups.items():
        item = dict(zip(group_keys, key))
        item["trajectories"] = len(values)
        for metric in metric_keys:
            observations = np.array([v[metric] for v in values])
            item[metric] = float(observations.mean())
            item[metric+"_std"] = float(observations.std(ddof=1)) if len(values)>1 else 0.
        result.append(item)
    return result


def run(output, train=6, validation=2, test=4, budget=256, duration=10.):
    output = Path(output)
    for folder in ("trajectories", "models"):
        (output/folder).mkdir(parents=True, exist_ok=True)
    physics, filter_cfg, sensor = PendulumConfig(), FilterConfig(), RamseyConfig()
    dt, horizon = .02, 5  # 100 ms prediction, 1 s free rollout
    rng = np.random.default_rng(20261008)
    initial_angles = rng.uniform(np.deg2rad(-50), np.deg2rad(50), train+validation+test)
    velocities = rng.uniform(-.4, .4, len(initial_angles))
    records, estimation_rows, prediction_rows = [], [], []
    for index, (angle, velocity) in enumerate(zip(initial_angles, velocities)):
        split = split_for(index, train, validation)
        for original in SCENARIOS:
            scenario = replace(original, duration=duration, dt=dt, seed=original.seed+10000*index)
            truth_physics = replace(physics, initial_angle=float(angle), initial_velocity=float(velocity))
            dataset = generate_dataset(scenario, truth_physics)
            estimates = {}
            for mode in MODES:
                scenario_sensor = replace(sensor, coherence_time=sensor.coherence_time/3
                                          if original.name == "high_noise" else sensor.coherence_time)
                result = estimate(dataset, scenario.measurement_noise_std, mode, physics,
                                  filter_cfg, scenario_sensor, seed=scenario.seed+500000)
                estimates[mode] = result
                if split == "test":
                    estimation_rows.append({"trajectory": index, "scenario": scenario.name,
                                            "estimator": mode,
                                            **estimation_metrics(dataset["true_state"], result)})
            name = f"trajectory_{index:02d}_{scenario.name}"
            payload = {key: value for key, value in dataset.items() if isinstance(value, np.ndarray)}
            for mode, result in estimates.items():
                payload.update({f"{mode}_{key}": value for key, value in result.items()})
            np.savez_compressed(output/"trajectories"/(name+".npz"), **payload)
            records.append({"id": index, "split": split, "scenario": scenario.name,
                            "file": name+".npz", "data": dataset, "estimates": estimates})
        print(f"Simulated trajectory {index+1}/{len(initial_angles)} ({split})", flush=True)

    models, selections = {}, {}
    for mode in ("classical", "ramsey_adaptive"):
        x, target = supervised_pairs(records, mode, "train", horizon, budget)
        val_x, val_target = supervised_pairs(records, mode, "validation", horizon, 160)
        for model_type in ("classical", "quantum"):
            model, scores = select_model(x, target, val_x, val_target, quantum=model_type == "quantum")
            models[mode, model_type] = model
            selections[f"{mode}_{model_type}"] = {"selected": asdict(model), "candidates": scores}
            model.save(output/"models"/f"{mode}_{model_type}.npz")
    for item in records:
        if item["split"] != "test":
            continue
        for (mode, model_type), model in models.items():
            metrics = prediction_metrics(item["data"]["true_state"], item["estimates"][mode]["state"],
                                         model, horizon, physics, dt,
                                         covariance=item["estimates"][mode]["covariance"])
            prediction_rows.append({"trajectory": item["id"], "scenario": item["scenario"],
                                    "estimator": mode, "model": model_type, **metrics})

    # F3 shot efficiency, each budget uses the same test truth and schedules.
    efficiency_rows = []
    nominal_test = [item for item in records if item["split"] == "test" and item["scenario"] == "nominal"]
    for shots in (64, 256, 1024):
        for mode in ("ramsey_fixed", "ramsey_adaptive"):
            for item in nominal_test:
                shot_sensor = replace(sensor, shots=shots)
                result = estimate(item["data"], SCENARIOS[0].measurement_noise_std, mode,
                                  physics, filter_cfg, shot_sensor, seed=700000+shots+item["id"])
                efficiency_rows.append({"shots": shots, "estimator": mode,
                                        **estimation_metrics(item["data"]["true_state"], result),
                                        "total_shots": int(result["used"].sum()*shots),
                                        "summed_spin_interrogation_s": float(np.nansum(result["tau"])*shots),
                                        "mean_fisher_information": float(np.nanmean(result["fisher_information"])),
                                        "mean_qfi": float(np.nanmean(result["qfi"]))})
    # F4/F5 sample efficiency on nominal test only, validation never changes.
    learning_rows = []
    for samples in (64, 128, budget):
        for mode in ("classical", "ramsey_adaptive"):
            x, target = supervised_pairs(records, mode, "train", horizon, samples)
            val_x, val_target = supervised_pairs(records, mode, "validation", horizon, 160)
            for model_type in ("classical", "quantum"):
                model = models[mode, model_type] if samples == budget else select_model(
                    x, target, val_x, val_target, quantum=model_type == "quantum")[0]
                for item in nominal_test:
                    metrics = prediction_metrics(item["data"]["true_state"], item["estimates"][mode]["state"],
                                                 model, horizon, physics, dt)
                    learning_rows.append({"samples": samples, "estimator": mode,
                                          "model": model_type, **metrics})
    # S2/S3 sequential measurement requests driven by EKF covariance/innovation.
    policy_rows = []
    for item in records:
        if item["split"] != "test":
            continue
        noise = next(s.measurement_noise_std for s in SCENARIOS if s.name == item["scenario"])
        for policy in ("always", "event"):
            result = (item["estimates"]["classical"] if policy == "always" else
                      estimate(item["data"], noise, physics=physics, cfg=filter_cfg, policy=policy))
            policy_rows.append({"scenario": item["scenario"], "policy": policy,
                                **estimation_metrics(item["data"]["true_state"], result)})
    # S4 OOD initial angles beyond training range and finite query noise ablation.
    robustness_rows = []
    for index in range(test):
        scenario = replace(SCENARIOS[0], dt=dt, duration=duration, seed=800000+index)
        dataset = generate_dataset(scenario, replace(physics, initial_angle=np.deg2rad(75+index*3),
                                                    initial_velocity=.2))
        for mode in ("classical", "ramsey_adaptive"):
            result = estimate(dataset, scenario.measurement_noise_std, mode, physics, filter_cfg,
                              sensor, seed=scenario.seed)
            for model_type in ("classical", "quantum"):
                metrics = prediction_metrics(dataset["true_state"], result["state"], models[mode, model_type],
                                             horizon, physics, dt)
                robustness_rows.append({"condition": "ood_75_to_84_deg", "estimator": mode,
                                        "model": model_type, **metrics})
    for depolarization in (0., .05, .2):
        for mode in ("classical", "ramsey_adaptive"):
            for item in nominal_test:
                metrics = prediction_metrics(item["data"]["true_state"], item["estimates"][mode]["state"],
                                             models[mode, "quantum"], horizon, physics, dt, depolarization)
                robustness_rows.append({"condition": f"query_depolarization_{depolarization}",
                                        "estimator": mode, "model": "quantum", **metrics})
    estimator_metrics = list(estimation_rows[0].keys())[3:]
    forecast_metrics = list(prediction_rows[0].keys())[4:]
    metadata = {"seed": 20261008, "python": platform.python_version(), "numpy": np.__version__,
                "dt_s": dt, "duration_s": duration, "prediction_horizon_s": dt*horizon,
                "train_trajectories": train, "validation_trajectories": validation,
                "test_trajectories": test, "training_pairs": budget,
                "physics": asdict(physics), "filter": asdict(filter_cfg), "ramsey": asdict(sensor),
                "split_unit": "entire initial-condition trajectory, shared across scenarios",
                "training_targets": "future estimator states minus current estimator states",
                "validation_targets": "held-out future estimator states, no test truth for tuning",
                "prediction_uncertainty": "J P J^T + validation residual covariance, approximate independence",
                "training_conditions": "nominal only", "quantum_backend": "exact NumPy statevector",
                "feature_circuit": {"qubits": 2, "single_qubit_gates": 6, "cnot_gates": 2,
                                     "kernel_training_pairs_upper_bound": budget*(budget+1)//2},
                "limitations": ["Illustrative field gradient and spin coherence, no validated NV device",
                                "Ideal parallel independent spin readout and frozen field within each batch",
                                "Classical Gaussian noise and Ramsey shots have unequal physical resource budgets",
                                "Two-qubit kernel is efficiently classically simulable, no speedup claim",
                                "Only four default test trajectories, preliminary descriptive statistics",
                                "Spring policy is a measurement heuristic, no control or safety proof"],
                "selections": selections}
    summary = {"metadata": metadata,
               "estimation": aggregate(estimation_rows, ["scenario", "estimator"], estimator_metrics),
               "prediction": aggregate(prediction_rows, ["scenario", "estimator", "model"], forecast_metrics),
               "shot_efficiency": aggregate(efficiency_rows, ["shots", "estimator"],
                                             ["angle_rmse_deg", "total_shots", "summed_spin_interrogation_s",
                                              "mean_fisher_information", "mean_qfi"]),
               "learning_curves": aggregate(learning_rows, ["samples", "estimator", "model"],
                                            [m for m in forecast_metrics if m != "prediction_angle_95_coverage"]),
               "measurement_policy": aggregate(policy_rows, ["scenario", "policy"], estimator_metrics),
               "robustness": aggregate(robustness_rows, ["condition", "estimator", "model"],
                                        [m for m in forecast_metrics if m != "prediction_angle_95_coverage"])}
    manifest = [{k: item[k] for k in ("id", "split", "scenario", "file")} for item in records]
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"per_trajectory.json").write_text(json.dumps({"estimation": estimation_rows,
        "prediction": prediction_rows, "shot_efficiency": efficiency_rows,
        "learning_curves": learning_rows, "measurement_policy": policy_rows,
        "robustness": robustness_rows}, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Results written to {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/progress"))
    parser.add_argument("--train", type=int, default=6)
    parser.add_argument("--validation", type=int, default=2)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--budget", type=int, default=256)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 1 or args.budget < 16:
        parser.error("each split needs a trajectory and budget must be at least 16")
    run(args.output, args.train, args.validation, args.test, args.budget)


if __name__ == "__main__":
    main()
