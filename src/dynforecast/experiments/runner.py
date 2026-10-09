"""Single-experiment runner and sequential benchmark sweeps.

``run_experiment`` goes through the same steps for every model:
prepare data -> fit -> forecast the evaluation origins -> score -> save. Each
run is written to ``<output>/<run_id>``, where the run id hashes the config,
data, code, and package versions so identical runs can be resumed.
"""

import copy
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import platform

try:
    import resource
except ImportError:  # not available on Windows
    resource = None
import sys
import time
import traceback

import filelock
import numpy as np
import torch

from dynforecast.data import Standardizer, chronological_bounds, windows, load_csv, download_dataset
from dynforecast.data.pipeline import corrupt
from dynforecast.simulations import simulate
from dynforecast.models import make_model, DYNAMICAL_MODELS, PRIOR_MODELS
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
ARTIFACTS = ("metrics.json", "predictions.npz", "model.pkl")
COPIED_DIAGNOSTICS = (
    "best_epoch",
    "stop_epoch",
    "peak_gpu_memory_bytes",
    "training_seconds",
    "solver_evaluations",
)


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


def _short_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _record_failure(folder, cfg, stage, exc):
    """Write a failed-run manifest and traceback for an error before the run started."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": folder.name,
        "status": "failed",
        "stage": stage,
        "config": cfg,
        "error": f"{type(exc).__name__}: {exc}",
    }
    atomic_json(folder / "manifest.json", manifest)
    atomic_json(folder / "config.json", cfg)
    (folder / "error.log").write_text(traceback.format_exc())
    return manifest


def run_identity(cfg, series):
    """Hash everything that determines a result: config, data, code, and packages."""
    identity = {
        k: v for k, v in cfg.items() if k not in ("output", "budget_seconds", "sweep", "profile")
    }
    identity["data_sha256"] = fingerprint(series)
    identity["code_sha256"] = code_fingerprint()
    identity["packages"] = environment_manifest()["packages"]
    return _short_hash(json.dumps(identity, sort_keys=True)), identity


def _completed_run(folder):
    """Return the manifest of a finished run in ``folder``, or None."""
    path = folder / "manifest.json"
    if not path.exists():
        return None
    previous = json.loads(path.read_text())
    if previous["status"] != "completed":
        return None
    if not all((folder / name).exists() for name in ARTIFACTS):
        raise RuntimeError("Completed run has missing artifacts; use --no-resume to rebuild")
    return previous


@dataclass
class PreparedData:
    """Standardized splits plus the bookkeeping needed to score and save a run."""

    inputs: list
    targets: list  # positions of the targets within ``inputs``
    target_columns: list  # the targets as original observation columns
    observations: np.ndarray
    scaler: Standardizer
    fitted: np.ndarray  # raw training rows the scaler was fitted on
    train: np.ndarray
    validation: np.ndarray
    clean: np.ndarray  # whole series, standardized, never corrupted
    corruption: dict
    validation_start: int
    test_start: int
    train_end: int
    dt: float


def select_columns(cfg, series):
    """Resolve and validate the input and target columns."""
    inputs = cfg["inputs"] if cfg["inputs"] is not None else list(range(len(series.names)))
    if len(set(inputs)) != len(inputs) or not inputs:
        raise ValueError("Input feature indices must be nonempty and unique")
    if any(not isinstance(i, int) or i < 0 or i >= len(series.names) for i in inputs):
        raise ValueError("Input indices must refer to existing observation columns")
    target_columns = cfg["targets"]
    if not target_columns or not all(t in inputs for t in target_columns):
        raise ValueError("Targets must be nonempty and included in observed inputs")
    return inputs, target_columns


def check_model_assumptions(cfg, series, inputs):
    """Reject data a model family cannot handle, before any fitting."""
    name = cfg["model"]["name"]
    if name in DYNAMICAL_MODELS:
        if series.inputs is not None:
            raise ValueError(
                "Autonomous dynamics models cannot ignore forcing; use temporal models"
            )
        if not np.allclose(np.diff(series.time), np.diff(series.time)[0], rtol=1e-5):
            raise ValueError("Dynamics currently require regular sampling")
        n_states = None if series.latent is None else series.latent.shape[1]
        full_state = (
            n_states is not None
            and len(inputs) == n_states
            and series.metadata.get("observed") == list(range(n_states))
        )
        if name != "latent_ode" and not full_state and not cfg["dynamics_assumption"]:
            raise ValueError("Partial/real observations require explicit dynamics_assumption=true")
    if name in PRIOR_MODELS and (inputs != [0, 1] or series.inputs is not None):
        raise ValueError(
            "Oscillator prior requires unforced, ordered position/velocity inputs [0,1]"
        )


def prepare_data(cfg, series):
    """Validate the run against the data and build standardized chronological splits."""
    inputs, target_columns = select_columns(cfg, series)
    observations = series.observations[:, inputs]
    validation_start, test_start = chronological_bounds(len(observations), **cfg["split"])
    fraction = cfg["train_fraction"]
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0,1]")
    train_end = int(validation_start * fraction)
    history = cfg["history"]
    if train_end < history + cfg["horizon"]:
        raise ValueError("Training subset too short; increase fraction or reduce windows")
    check_model_assumptions(cfg, series, inputs)

    corruption = dict(cfg["corruption"])
    protocol = corruption.pop("protocol", "test_time")
    if protocol not in ("test_time", "robust_training"):
        raise ValueError("Corruption protocol must be test_time or robust_training")
    fitted = observations[:train_end].copy()
    if protocol == "robust_training":
        fitted = corrupt(fitted, cfg["seed"] + 1000, **corruption)
    scaler = Standardizer().fit(fitted)
    clean = scaler.transform(observations)
    return PreparedData(
        inputs=inputs,
        targets=[inputs.index(t) for t in target_columns],
        target_columns=target_columns,
        observations=observations,
        scaler=scaler,
        fitted=fitted,
        train=scaler.transform(fitted),
        # Validation windows start one history before the split so targets stay inside it.
        validation=clean[validation_start - history : test_start],
        clean=clean,
        corruption=corruption,
        validation_start=validation_start,
        test_start=test_start,
        train_end=train_end,
        dt=float(np.diff(series.time)[0]),
    )


def fit_model(cfg, data, folder, resume, deadline):
    """Build the configured model and fit it, passing the options its family needs."""
    model = make_model(cfg["model"]["name"], cfg["model"], cfg["seed"])
    validation = data.validation
    options = {}
    if isinstance(model, (NeuralForecaster, NeuralODE, HybridForecaster)):
        options = {"deadline": deadline, "checkpoint": folder / "checkpoint.pt", "resume": resume}
        if cfg.get("tracking", {}).get("tensorboard"):
            options["tensorboard_dir"] = folder / "tensorboard"
    if isinstance(model, (NeuralODE, MechanisticOscillator, HybridForecaster)):
        options.update(mean=data.scaler.mean, scale=data.scaler.scale)
    if isinstance(model, NeuralODE):
        # ODEs fit trajectories, so they validate on the split itself, without history.
        validation = data.clean[data.validation_start : data.test_start]
    model.fit(
        data.train, validation, data.targets, cfg["history"], cfg["horizon"], data.dt, **options
    )
    return model


@dataclass
class Forecasts:
    split: str
    start: int
    origins: np.ndarray
    histories: np.ndarray  # standardized model inputs, possibly corrupted
    truth: np.ndarray
    predictions: np.ndarray
    inference_seconds: float


def forecast(cfg, data, model):
    """Forecast every evaluation origin and return predictions in original units."""
    split = cfg.get("evaluation", "test")
    if split not in ("test", "validation"):
        raise ValueError("Evaluation must be test or validation")
    history, horizon, stride = cfg["history"], cfg["horizon"], cfg["stride"]
    start, stop = (
        (data.validation_start, data.test_start)
        if split == "validation"
        else (data.test_start, len(data.clean))
    )
    _, _, origins = windows(
        data.clean, history, horizon, data.targets, start=start, stop=stop, stride=stride
    )
    # Corruption only touches the model's inputs, never the targets or the training data.
    observed = data.observations.copy()
    observed[start - history :] = corrupt(
        observed[start - history :], cfg["seed"] + 2000, **data.corruption
    )
    histories, _, _ = windows(
        data.scaler.transform(observed),
        history,
        horizon,
        data.targets,
        start=start,
        stop=stop,
        stride=stride,
    )
    truth = np.stack([data.observations[o : o + horizon, data.targets] for o in origins])
    if not np.isfinite(truth).all():
        raise ValueError(
            "Evaluation targets contain missing values; choose a complete target period"
        )
    tick = time.perf_counter()
    predictions = check_predictions(model.predict(histories), truth.shape)
    inference_seconds = time.perf_counter() - tick
    return Forecasts(
        split=split,
        start=start,
        origins=origins,
        histories=histories,
        truth=truth,
        predictions=data.scaler.inverse(predictions, data.targets),
        inference_seconds=inference_seconds,
    )


def _peak_memory_bytes():
    if resource is None:
        return None
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def _count_flops(model, histories):
    """FLOPs for one forecast origin, counted by the PyTorch profiler."""
    diagnostics = copy.deepcopy(model.diagnostics)  # predict may overwrite them
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as profiler:
        model.predict(histories[:1])
    model.diagnostics = diagnostics
    return sum(event.flops for event in profiler.key_averages())


def score(cfg, series, data, model, result, fit_seconds):
    """Forecast accuracy plus cost, capacity, and model-specific diagnostics."""
    metrics = forecast_metrics(
        result.truth,
        result.predictions,
        data.scaler.impute(data.fitted)[:, data.targets],
        cfg.get("seasonality", 1),
    )
    metrics.update(
        fit_seconds=fit_seconds,
        inference_seconds=result.inference_seconds,
        latency_per_origin_seconds=result.inference_seconds / len(result.origins),
        parameter_count=int(model.parameters),
        n_training_observations=data.train_end,
        n_training_trajectories=1,
        n_test_origins=len(result.origins),
        process_peak_rss_bytes=_peak_memory_bytes(),
        memory_scope="whole-process high-water mark, not incremental model memory",
        rollout=model.rollout,
        physical_horizon=cfg["horizon"] * data.dt,
    )
    if isinstance(model, SINDy):
        metrics.update(equation_metrics(model, series, data.scaler, data.inputs, result.start))
    metrics["evaluation_split"] = result.split
    for key in COPIED_DIAGNOSTICS:
        if key in (model.diagnostics or {}):
            metrics[key] = model.diagnostics[key]
    if cfg.get("profile_flops") and hasattr(model, "network"):
        metrics["supported_operation_flops_per_origin"] = _count_flops(model, result.histories)
        metrics["flop_scope"] = (
            "PyTorch-supported matmul/convolution operations only; excludes solver/control overhead"
        )
    metrics["prediction_stability"] = {
        "finite_fraction": 1.0,
        "max_absolute_value": float(np.max(np.abs(result.predictions))),
    }
    if cfg.get("dynamical_metrics"):
        from dynforecast.evaluation.dynamics import dynamical_metrics

        metrics["dynamical_fidelity"] = dynamical_metrics(result.truth, result.predictions, data.dt)
    return metrics


def save_artifacts(folder, cfg, series, data, model, result, metrics):
    atomic_json(folder / "metrics.json", metrics)
    atomic_json(folder / "diagnostics.json", model.diagnostics or {})
    atomic_json(
        folder / "preprocessing.json",
        {
            "mean": data.scaler.mean.tolist(),
            "scale": data.scaler.scale.tolist(),
            "inputs": data.inputs,
            "targets": data.targets,
            "train_end": data.train_end,
            "validation_start": data.validation_start,
            "test_start": data.test_start,
            "imputation": "causal forward fill",
        },
    )
    history, horizon, origins = cfg["history"], cfg["horizon"], result.origins
    np.savez_compressed(
        folder / "predictions.npz",
        truth=result.truth,
        predictions=result.predictions,
        origins=origins,
        time=series.time[origins],
        target_times=np.stack([series.time[o : o + horizon] for o in origins]),
        history=np.stack([data.observations[o - history : o, data.targets] for o in origins]),
        history_times=np.stack([series.time[o - history : o] for o in origins]),
        target_names=np.array([series.names[t] for t in data.target_columns]),
    )
    with (folder / "model.tmp").open("wb") as f:
        pickle.dump(model, f)
    os.replace(folder / "model.tmp", folder / "model.pkl")
    if cfg.get("tracking", {}).get("mlflow"):
        from dynforecast.training.tracking import log_mlflow

        log_mlflow(folder, cfg, metrics)


def run_experiment(configuration, resume=True, deadline=None):
    """Run one configured experiment and write its artifacts to ``output/<run_id>``.

    An identical completed run is reused when ``resume`` is true. Returns the run
    manifest, with ``status`` set to ``completed``, ``failed``, or ``interrupted``.
    """
    cfg = merge(DEFAULT, configuration)
    cfg.pop("sweep", None)
    if cfg["model"].get("device") == "auto":
        cfg["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(cfg["output"])

    if cfg["dataset"].get("trajectories") or cfg["dataset"]["name"] == "monash":
        from .collection import run_collection

        try:
            return run_collection(cfg, resume, deadline)
        except Exception as exc:
            folder = output / _short_hash(json.dumps(cfg, sort_keys=True))
            return _record_failure(folder, cfg, "collection", exc)

    seed_everything(cfg["seed"])
    try:
        series = load_series(cfg["dataset"])
    except Exception as exc:
        identity = {k: v for k, v in cfg.items() if k != "output"}
        folder = output / _short_hash(json.dumps(identity, sort_keys=True) + ":ingestion")
        return _record_failure(folder, cfg, "ingestion", exc)

    run_id, identity = run_identity(cfg, series)
    folder = output / run_id
    folder.mkdir(parents=True, exist_ok=True)
    if resume and (previous := _completed_run(folder)):
        return previous

    try:
        lock = filelock.FileLock(str(folder / ".lock"), timeout=0)
        lock.acquire()
    except filelock.Timeout:
        raise RuntimeError(f"Run {run_id} is already running in another process") from None

    manifest_path = folder / "manifest.json"
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
        data = prepare_data(cfg, series)
        model = fit_model(cfg, data, folder, resume, deadline)
        fit_seconds = time.perf_counter() - tick
        result = forecast(cfg, data, model)
        metrics = score(cfg, series, data, model, result, fit_seconds)
        save_artifacts(folder, cfg, series, data, model, result, metrics)
        manifest.update(status="completed", metrics=metrics)
    except BaseException as exc:
        interrupted = isinstance(exc, (TimeoutError, KeyboardInterrupt))
        manifest.update(
            status="interrupted" if interrupted else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        (folder / "error.log").write_text(traceback.format_exc())
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - tick
        atomic_json(manifest_path, manifest)
        lock.release()
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
