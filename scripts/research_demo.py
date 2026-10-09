"""Run a bounded, reproducible end-to-end demonstration and generate offline reports.

This is a development study, not final model-comparison evidence. Default runtime
budget is ten minutes on CPU; interrupted configurations can resume on the next run.
"""

import argparse
import copy
import json
from pathlib import Path
import time
from dynforecast.experiments import run_experiment
from dynforecast.experiments.runner import atomic_json
from dynforecast.models import MODEL_NAMES
from dynforecast.visualization import generate_report
from dynforecast.visualization.dashboard import generate_dashboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/research_demo")
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    deadline = time.monotonic() + args.budget_seconds
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    base = {
        "dataset": {
            "name": "oscillator",
            "n": 700,
            "dt": 0.05,
            "seed": 17,
            "params": {"damping": 0.03},
        },
        "model": {"epochs": args.epochs, "width": 16, "patience": 6, "trees": 12, "period": 12},
        "history": 32,
        "horizon": 24,
        "targets": [0, 1],
        "stride": 12,
        "dynamical_metrics": True,
        "output": str(root),
    }
    configurations = []
    # Representative model-family coverage, followed by repeated-seed matched sweeps.
    for name in MODEL_NAMES:
        cfg = copy.deepcopy(base)
        cfg["model"]["name"] = name
        if name.startswith("nf_"):
            cfg.update(inputs=[0], targets=[0])
            cfg["model"]["max_steps"] = 15
        if name == "arima":
            # Stochastic AR data satisfy ARIMA's innovation assumptions; a noise-free
            # exact oscillator can make maximum-likelihood variance estimation singular.
            cfg["dataset"] = {"name": "ar", "n": 700, "seed": 17}
            cfg["targets"] = [0]
        configurations.append(cfg)
    # Match ARIMA's stochastic AR task with the same-data statistical/ML/neural baselines.
    for name in ["persistence", "ar", "var", "ridge", "lstm"]:
        cfg = copy.deepcopy(base)
        cfg.update(dataset={"name": "ar", "n": 700, "seed": 17}, targets=[0])
        cfg["model"]["name"] = name
        configurations.append(cfg)
    for seed in [0, 1, 2]:
        for name in ["persistence", "var", "lstm", "sindy", "hybrid"]:
            for horizon in [6, 24, 60]:
                cfg = copy.deepcopy(base)
                cfg.update(seed=seed, horizon=horizon)
                cfg["model"]["name"] = name
                configurations.append(cfg)
            for fraction in [0.25, 0.5, 1.0]:
                cfg = copy.deepcopy(base)
                cfg.update(seed=seed, train_fraction=fraction)
                cfg["model"]["name"] = name
                configurations.append(cfg)
            for protocol in ["test_time", "robust_training"]:
                for noise in [0.0, 0.05, 0.15]:
                    cfg = copy.deepcopy(base)
                    cfg.update(seed=seed, corruption={"protocol": protocol, "noise": noise})
                    cfg["model"]["name"] = name
                    configurations.append(cfg)
    for name in ["persistence", "var", "lstm", "sindy", "hybrid", "latent_ode"]:
        cfg = copy.deepcopy(base)
        cfg["dataset"].update(n=200, trajectories=5)
        cfg["model"]["name"] = name
        configurations.append(cfg)
    # Keep target position fixed while changing only the available observations.
    for name in ["persistence", "ridge", "lstm"]:
        for inputs in [[0], [0, 1]]:
            cfg = copy.deepcopy(base)
            cfg.update(inputs=inputs, targets=[0])
            cfg["model"]["name"] = name
            configurations.append(cfg)
    # Include systems beyond the oscillator, including forcing and chaotic dynamics.
    for dataset in ["vanderpol", "lotka_volterra", "lorenz", "duffing", "arx"]:
        for name in ["persistence", "ridge", "lstm"]:
            cfg = copy.deepcopy(base)
            cfg.update(dataset={"name": dataset, "n": 500, "dt": 0.03}, targets=[0])
            cfg["model"]["name"] = name
            configurations.append(cfg)
    for name in ["mechanistic", "hybrid", "adaptive_hybrid"]:
        for stiffness in [1.0, 2.0]:
            cfg = copy.deepcopy(base)
            cfg["model"].update(name=name, prior={"stiffness": stiffness, "damping": 0.1})
            configurations.append(cfg)
    for inputs in [[0], [0, 1]]:
        cfg = copy.deepcopy(base)
        cfg.update(inputs=inputs, targets=[0])
        cfg["model"]["name"] = "latent_ode"
        configurations.append(cfg)
    # Actual environmental measurements, including alternatives to blocked Jena acquisition.
    for dataset, targets in [("ett", [6]), ("weather", [0]), ("co2", [0]), ("macrodata", [0])]:
        for model in ["persistence", "ridge", "lstm"]:
            cfg = copy.deepcopy(base)
            cfg["dataset"] = {"name": dataset, "max_rows": 1800}
            cfg.update(targets=targets, history=24, horizon=12, stride=12)
            cfg["model"]["name"] = model
            cfg["comparison_tag"] = "real_measurements"
            configurations.append(cfg)
    # Deliberate stress cases and spatial dynamics.
    for extra in [
        {"corruption": {"protocol": "test_time", "missing": 0.15}},
        {"corruption": {"protocol": "test_time", "outage": 0.2}},
        {"corruption": {"protocol": "test_time", "outliers": 0.03}},
        {"corruption": {"protocol": "test_time", "bias": 0.1}},
        {"dataset": {"name": "oscillator", "n": 700, "irregular": 0.3}},
        {
            "dataset": {
                "name": "oscillator",
                "n": 700,
                "drift": {"at": 28.0, "params": {"stiffness": 2.0}},
            }
        },
        {"dataset": {"name": "reaction_diffusion", "n": 300, "grid": 8}, "targets": [0, 1]},
    ]:
        cfg = copy.deepcopy(base)
        cfg.update(extra)
        cfg["model"]["name"] = "lstm"
        configurations.append(cfg)
    outcomes = []
    for index, cfg in enumerate(configurations):
        if time.monotonic() >= deadline:
            break
        result = run_experiment(cfg, deadline=deadline)
        outcomes.append(result)
        print(
            f"[{index + 1}/{len(configurations)}] {cfg['model']['name']} / {cfg['dataset']['name']}: {result['status']}",
            flush=True,
        )
    # Preserve failures and operational limits; report only recorded results.
    summary = {
        "requested": len(configurations),
        "processed": len(outcomes),
        "completed": sum(o["status"] == "completed" for o in outcomes),
        "failed": sum(o["status"] == "failed" for o in outcomes),
        "interrupted": sum(o["status"] == "interrupted" for o in outcomes),
        "unlaunched": len(configurations) - len(outcomes),
        "budget_seconds": args.budget_seconds,
    }
    atomic_json(root / "demonstration.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(generate_report(root), flush=True)
    print(generate_dashboard(root), flush=True)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
