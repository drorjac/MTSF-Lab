import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import platform

try:
    import resource
except ImportError:  # Windows desktop environments do not expose POSIX rusage.
    resource = None
import sys
import time
import traceback
import numpy as np
import torch
from dynforecast.data import Standardizer, chronological_bounds, windows, load_csv, download_dataset
from dynforecast.data.pipeline import corrupt
from dynforecast.simulations import simulate
from dynforecast.models import make_model
from dynforecast.models.base import check_predictions
from dynforecast.models.dynamical import SINDy, NeuralODE
from dynforecast.models.neural import NeuralForecaster
from dynforecast.models.mechanistic import MechanisticOscillator
from dynforecast.models.hybrid import HybridForecaster
from dynforecast.training import seed_everything
from dynforecast.evaluation import forecast_metrics, equation_metrics


DEFAULT = {
    "dataset": {"name": "oscillator", "n": 600, "dt": 0.05, "seed": 0},
    "model": {"name": "persistence", "epochs": 30, "width": 32, "device": "cpu"},
    "history": 24,
    "horizon": 12,
    "targets": [0],
    "inputs": None,
    "seed": 0,
    "train_fraction": 1.0,
    "split": {"train": 0.6, "validation": 0.2},
    "stride": 4,
    "corruption": {"protocol": "test_time", "noise": 0.0, "missing": 0.0},
    "output": "results",
    "dynamics_assumption": False,
}
DYNAMICAL_NAMES = {
    "sindy",
    "pysindy",
    "neural_ode",
    "hybrid",
    "mechanistic",
    "sindy_hybrid",
    "trainable_mechanistic",
}
DYNAMICAL_NAMES.add("latent_ode")
PRIOR_NAMES = {
    "hybrid",
    "mechanistic",
    "trainable_mechanistic",
    "forecast_residual",
    "adaptive_hybrid",
    "average_hybrid",
    "physics_guided",
}


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def load_series(config):
    cfg = dict(config)
    name = cfg.pop("name")
    include_inputs = cfg.pop("include_inputs", False)
    if name in ("co2", "sunspots", "macrodata"):
        from dynforecast.data.builtin import load_builtin

        return load_builtin(name, cfg.get("max_rows"), cfg.get("columns"))
    if name in ("ett", "jena", "weather", "csv"):
        # Synthetic defaults remain in the merged config; they are not CSV loader options.
        for key in ("n", "dt", "seed", "noise", "observed", "initial", "params"):
            cfg.pop(key, None)
        path = cfg.pop("path", None)
        automatic = path is None
        if path is None:
            path = download_dataset(name, cfg.pop("directory", "data/raw"))
        if name == "jena":
            cfg.setdefault("missing_values", {"wv (m/s)": -9999.0, "max. wv (m/s)": -9999.0})
        series = load_csv(path, **cfg)
        if automatic:
            series.metadata["acquisition"] = json.loads(
                Path(path).with_suffix(".provenance.json").read_text()
            )
        return series
    series = simulate(system=name, **cfg)
    if include_inputs and series.inputs is not None:
        series.observations = np.column_stack([series.observations, series.inputs])
        series.names.extend([f"u{i}" for i in range(series.inputs.shape[1])])
        series.metadata["input_availability"] = (
            "historical observations only; no future forcing supplied"
        )
    return series


def fingerprint(series):
    digest = hashlib.sha256()
    digest.update(series.time.tobytes())
    digest.update(series.observations.tobytes())
    digest.update(json.dumps(series.names).encode())
    return digest.hexdigest()


def code_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_manifest():
    import importlib.metadata

    packages = {
        p: importlib.metadata.version(p)
        for p in (
            "dynforecastlab",
            "numpy",
            "scipy",
            "pandas",
            "torch",
            "torchdiffeq",
            "scikit-learn",
            "statsmodels",
            "PyYAML",
        )
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
    }


def run_experiment(configuration, resume=True, deadline=None):
    cfg = merge(DEFAULT, configuration)
    cfg.pop("sweep", None)
    if cfg["model"].get("device") == "auto":
        cfg["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["dataset"].get("trajectories") or cfg["dataset"]["name"] == "monash":
        from .collection import run_collection

        try:
            return run_collection(cfg, resume, deadline)
        except Exception as exc:
            path = (
                Path(cfg["output"])
                / hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
            )
            path.mkdir(parents=True, exist_ok=True)
            result = {
                "run_id": path.name,
                "status": "failed",
                "stage": "collection",
                "config": cfg,
                "error": f"{type(exc).__name__}: {exc}",
            }
            atomic_json(path / "manifest.json", result)
            (path / "error.log").write_text(traceback.format_exc())
            return result
    seed_everything(cfg["seed"])
    try:
        series = load_series(cfg["dataset"])
    except Exception as exc:
        identity = {k: v for k, v in cfg.items() if k != "output"}
        run_id = hashlib.sha256(
            (json.dumps(identity, sort_keys=True) + ":ingestion").encode()
        ).hexdigest()[:16]
        folder = Path(cfg["output"]) / run_id
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "status": "failed",
            "stage": "ingestion",
            "config": cfg,
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(folder / "manifest.json", manifest)
        atomic_json(folder / "config.json", cfg)
        (folder / "error.log").write_text(traceback.format_exc())
        return manifest
    identity = {
        k: v for k, v in cfg.items() if k not in ("output", "budget_seconds", "sweep", "profile")
    }
    identity["data_sha256"] = fingerprint(series)
    identity["code_sha256"] = code_fingerprint()
    identity["packages"] = environment_manifest()["packages"]
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cfg["output"]) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["status"] == "completed":
            if all((folder / f).exists() for f in ("metrics.json", "predictions.npz", "model.pkl")):
                return previous
            raise RuntimeError("Completed run has missing artifacts; use --no-resume to rebuild")
    # Exclusive process lock prevents accidental concurrent writes to identical run IDs.
    lock = folder / ".lock"
    if lock.exists():
        try:
            owner = int(lock.read_text())
            os.kill(owner, 0)
        except ProcessLookupError:
            lock.unlink(missing_ok=True)
        except (ValueError, PermissionError):
            raise RuntimeError(f"Cannot establish lock owner; inspect {lock}") from None
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Run {run_id} is locked; inspect recorded PID before clearing {lock}"
        ) from None
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "config": cfg,
        "data_sha256": identity["data_sha256"],
        "code_sha256": identity["code_sha256"],
        "provenance": series.metadata,
        "environment": environment_manifest(),
    }
    atomic_json(folder / "config.json", cfg)
    atomic_json(manifest_path, manifest)
    tick = time.perf_counter()
    try:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Budget exhausted before fit")
        inputs = cfg["inputs"] if cfg["inputs"] is not None else list(range(len(series.names)))
        if len(set(inputs)) != len(inputs) or not inputs:
            raise ValueError("Input feature indices must be nonempty and unique")
        if any(not isinstance(i, int) or i < 0 or i >= len(series.names) for i in inputs):
            raise ValueError("Input indices must refer to existing observation columns")
        targets_original = cfg["targets"]
        if not targets_original or not all(t in inputs for t in targets_original):
            raise ValueError("Targets must be nonempty and included in observed inputs")
        targets = [inputs.index(t) for t in targets_original]
        observations = series.observations[:, inputs]
        a, b = chronological_bounds(len(observations), **cfg["split"])
        fraction = cfg["train_fraction"]
        if not 0 < fraction <= 1:
            raise ValueError("Training fraction must be in (0,1]")
        ntrain = int(a * fraction)
        history, horizon = cfg["history"], cfg["horizon"]
        if ntrain < history + horizon:
            raise ValueError("Training subset too short; increase fraction or reduce windows")
        name = cfg["model"]["name"]
        dynamical = name in DYNAMICAL_NAMES
        if dynamical:
            if series.inputs is not None:
                raise ValueError(
                    "Autonomous dynamics models cannot ignore forcing; use temporal models"
                )
            if not np.allclose(np.diff(series.time), np.diff(series.time)[0], rtol=1e-5):
                raise ValueError("Dynamics currently require regular sampling")
            observed = series.metadata.get("observed")
            full_state = (
                series.latent is not None
                and len(inputs) == series.latent.shape[1]
                and observed == list(range(series.latent.shape[1]))
            )
            if name != "latent_ode" and not full_state and not cfg["dynamics_assumption"]:
                raise ValueError(
                    "Partial/real observations require explicit dynamics_assumption=true"
                )
            if name in PRIOR_NAMES and inputs != [0, 1]:
                raise ValueError(
                    "Oscillator prior requires ordered measured position, velocity inputs [0,1]"
                )
        corruption = dict(cfg["corruption"])
        protocol = corruption.pop("protocol", "test_time")
        if protocol not in ("test_time", "robust_training"):
            raise ValueError("Corruption protocol must be test_time or robust_training")
        fitted = observations[:ntrain].copy()
        if protocol == "robust_training":
            fitted = corrupt(fitted, cfg["seed"] + 1000, **corruption)
        scaler = Standardizer().fit(fitted)
        train = scaler.transform(fitted)
        clean = scaler.transform(observations)
        # Validation targets are fixed, and past context ends exactly at the split boundary.
        validation = clean[a - history : b]
        dt = float(np.diff(series.time)[0])
        model = make_model(name, cfg["model"], cfg["seed"])
        options = {}
        if name in PRIOR_NAMES and (inputs != [0, 1] or series.inputs is not None):
            raise ValueError(
                "Oscillator-guided forecasting requires unforced position/velocity inputs [0,1]"
            )
        if isinstance(model, (NeuralForecaster, NeuralODE, HybridForecaster)):
            options = {
                "deadline": deadline,
                "checkpoint": folder / "checkpoint.pt",
                "resume": resume,
            }
            if cfg.get("tracking", {}).get("tensorboard"):
                options["tensorboard_dir"] = folder / "tensorboard"
        if isinstance(model, NeuralODE):
            validation = clean[a:b]
            options.update(mean=scaler.mean, scale=scaler.scale)
        elif isinstance(model, (MechanisticOscillator, HybridForecaster)):
            options.update(mean=scaler.mean, scale=scaler.scale)
        model.fit(train, validation, targets, history, horizon, dt, **options)
        fit_seconds = time.perf_counter() - tick
        evaluation = cfg.get("evaluation", "test")
        if evaluation not in ("test", "validation"):
            raise ValueError("Evaluation must be test or validation")
        start, stop = (a, b) if evaluation == "validation" else (b, len(clean))
        _, _, origins = windows(
            clean, history, horizon, targets, start=start, stop=stop, stride=cfg["stride"]
        )
        # Test-only corruption never changes ground-truth targets or training preprocessing.
        history_values = observations.copy()
        history_values[start - history :] = corrupt(
            history_values[start - history :], cfg["seed"] + 2000, **corruption
        )
        x, _, _ = windows(
            scaler.transform(history_values),
            history,
            horizon,
            targets,
            start=start,
            stop=stop,
            stride=cfg["stride"],
        )
        truth = np.stack([observations[o : o + horizon, targets] for o in origins])
        if not np.isfinite(truth).all():
            raise ValueError(
                "Evaluation targets contain missing values; choose a complete target period"
            )
        before = time.perf_counter()
        predictions = check_predictions(model.predict(x), truth.shape)
        inference = time.perf_counter() - before
        predictions = scaler.inverse(predictions, targets)
        metrics = forecast_metrics(
            truth, predictions, scaler.impute(fitted)[:, targets], cfg.get("seasonality", 1)
        )
        metrics.update(
            {
                "fit_seconds": fit_seconds,
                "inference_seconds": inference,
                "latency_per_origin_seconds": inference / len(origins),
                "parameter_count": int(model.parameters),
                "n_training_observations": ntrain,
                "n_training_trajectories": 1,
                "n_test_origins": len(origins),
                "process_peak_rss_bytes": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024)
                )
                if resource is not None
                else None,
                "memory_scope": "whole-process high-water mark, not incremental model memory",
                "rollout": model.rollout,
                "physical_horizon": horizon * dt,
            }
        )
        if isinstance(model, SINDy):
            metrics.update(equation_metrics(model, series, scaler, inputs, start))
        metrics["evaluation_split"] = evaluation
        if model.diagnostics:
            for key in (
                "best_epoch",
                "stop_epoch",
                "peak_gpu_memory_bytes",
                "training_seconds",
                "solver_evaluations",
            ):
                if key in model.diagnostics:
                    metrics[key] = model.diagnostics[key]
        if cfg.get("profile_flops") and hasattr(model, "network"):
            saved_diagnostics = copy.deepcopy(model.diagnostics)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
            ) as profiler:
                model.predict(x[:1])
            metrics["supported_operation_flops_per_origin"] = sum(
                event.flops for event in profiler.key_averages()
            )
            metrics["flop_scope"] = (
                "PyTorch-supported matmul/convolution operations only; excludes solver/control overhead"
            )
            model.diagnostics = saved_diagnostics
        metrics["prediction_stability"] = {
            "finite_fraction": 1.0,
            "max_absolute_value": float(np.max(np.abs(predictions))),
        }
        if cfg.get("dynamical_metrics"):
            from dynforecast.evaluation.dynamics import dynamical_metrics

            metrics["dynamical_fidelity"] = dynamical_metrics(truth, predictions, dt)
        atomic_json(folder / "metrics.json", metrics)
        atomic_json(folder / "diagnostics.json", model.diagnostics or {})
        atomic_json(
            folder / "preprocessing.json",
            {
                "mean": scaler.mean.tolist(),
                "scale": scaler.scale.tolist(),
                "inputs": inputs,
                "targets": targets,
                "train_end": ntrain,
                "validation_start": a,
                "test_start": b,
                "imputation": "causal forward fill",
            },
        )
        np.savez_compressed(
            folder / "predictions.npz",
            truth=truth,
            predictions=predictions,
            origins=origins,
            time=series.time[origins],
            target_times=np.stack([series.time[o : o + horizon] for o in origins]),
            history=np.stack([observations[o - history : o, targets] for o in origins]),
            history_times=np.stack([series.time[o - history : o] for o in origins]),
            target_names=np.array([series.names[t] for t in targets_original]),
        )
        with (folder / "model.tmp").open("wb") as f:
            pickle.dump(model, f)
        os.replace(folder / "model.tmp", folder / "model.pkl")
        if cfg.get("tracking", {}).get("mlflow"):
            from dynforecast.training.tracking import log_mlflow

            log_mlflow(folder, cfg, metrics)
        manifest.update(
            status="completed", metrics=metrics, elapsed_seconds=time.perf_counter() - tick
        )
    except BaseException as exc:
        manifest.update(
            status="interrupted"
            if isinstance(exc, (TimeoutError, KeyboardInterrupt))
            else "failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - tick,
        )
        (folder / "error.log").write_text(traceback.format_exc())
        atomic_json(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        atomic_json(manifest_path, manifest)
        lock.unlink(missing_ok=True)
    return manifest


def set_nested(config, key, value):
    parts = key.split(".")
    obj = config
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value


def run_benchmark(configuration, resume=True):
    """Sequential device-aware execution. Budget is a cooperative cutoff, not a hard kill."""
    cfg = merge(DEFAULT, configuration)
    if cfg.get("scheduler", {}).get("isolated"):
        from .scheduler import scheduled_benchmark

        return scheduled_benchmark(cfg, resume)
    budget = cfg.get("budget_seconds", 60)
    if budget <= 0:
        raise ValueError("Budget must be positive")
    deadline = time.monotonic() + budget
    sweep = cfg.get("sweep", {"model.name": ["persistence", "ar", "lstm", "sindy"]})
    if not sweep or any(not isinstance(v, list) or not v for v in sweep.values()):
        raise ValueError("Sweep values must be nonempty lists")
    results = []
    combinations = list(itertools.product(*sweep.values()))
    for combination in combinations:
        if time.monotonic() >= deadline:
            break
        current = copy.deepcopy(cfg)
        current.pop("sweep", None)
        for key, value in zip(sweep, combination):
            set_nested(current, key, value)
        outcome = run_experiment(current, resume, deadline)
        results.append(outcome)
        print(f"{outcome['run_id']}: {current['model']['name']} {outcome['status']}", flush=True)
    summary = {
        "requested": len(combinations),
        "processed": len(results),
        "completed": sum(r["status"] == "completed" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "interrupted": sum(r["status"] == "interrupted" for r in results),
        "budget_seconds": budget,
        "unlaunched": len(combinations) - len(results),
        "run_ids": [r["run_id"] for r in results],
    }
    Path(cfg["output"]).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(cfg["output"]) / "benchmark.json", summary)
    return summary
