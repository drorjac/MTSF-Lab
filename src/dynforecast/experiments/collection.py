"""Unseen-trajectory experiments. Windowing and derivatives never join independent series."""

import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
from dynforecast.data import Standardizer, load_tsf
from dynforecast.data.pipeline import corrupt, windows
from dynforecast.simulations import SYSTEMS, simulate
from dynforecast.models import make_model
from dynforecast.models.neural import NeuralForecaster
from dynforecast.models.dynamical import NeuralODE
from dynforecast.models.hybrid import HybridForecaster
from dynforecast.models.mechanistic import MechanisticOscillator
from dynforecast.models.base import check_predictions
from dynforecast.training import seed_everything
from dynforecast.evaluation import forecast_metrics


def make_collection(config):
    cfg = dict(config)
    name = cfg.pop("name")
    if name == "monash":
        path = cfg.get("path")
        if path is None and cfg.get("url"):
            from dynforecast.data.collections import download_tsf

            path = download_tsf(cfg["url"], cfg.get("directory", "data/monash"), cfg.get("member"))
        if not path:
            raise ValueError(
                "Monash collections require dataset.path to a .tsf file or dataset.url to an accessible source"
            )
        result = load_tsf(path)
        for index, series in enumerate(result):
            series.metadata.update(source=str(path), trajectory_id=index)
        return result
    count = cfg.pop("trajectories", 5)
    cfg.pop("collection_split", None)
    initials = cfg.pop("initial_conditions", None)
    parameter_sets = cfg.pop("parameter_sets", None)
    seed = cfg.get("seed", 0)
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        options = dict(cfg)
        options["seed"] = seed + i
        if initials:
            if len(initials) != count:
                raise ValueError("Provide one initial condition per trajectory")
            options["initial"] = initials[i]
        elif name in SYSTEMS:
            base = np.array(SYSTEMS[name]["initial"])
            # Additive perturbations vary initially zero velocity/state components too.
            options["initial"] = (base + rng.normal(0, 0.3, len(base))).tolist()
        if parameter_sets:
            if len(parameter_sets) != count:
                raise ValueError("Provide one parameter set per trajectory")
            options["params"] = {**options.get("params", {}), **parameter_sets[i]}
        series = simulate(name, **options)
        series.metadata["trajectory_id"] = i
        result.append(series)
    return result


def run_collection(cfg, resume=True, deadline=None):
    from .runner import (
        atomic_json,
        code_fingerprint,
        fingerprint,
        environment_manifest,
        DYNAMICAL_NAMES,
        PRIOR_NAMES,
    )

    seed_everything(cfg["seed"])
    collection = make_collection(cfg["dataset"])
    if len(collection) < 3:
        raise ValueError(
            "Trajectory-level experiments need at least train, validation and test trajectories"
        )
    identity = {k: v for k, v in cfg.items() if k not in ("output", "budget_seconds", "profile")}
    data_hash = hashlib.sha256("".join(fingerprint(s) for s in collection).encode()).hexdigest()
    identity.update(
        data_sha256=data_hash,
        code_sha256=code_fingerprint(),
        packages=environment_manifest()["packages"],
    )
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cfg["output"]) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["status"] == "completed" and all(
            (folder / name).exists()
            for name in ("metrics.json", "predictions.npz", "diagnostics.json", "model.pkl")
        ):
            return previous
    import filelock

    manifest = {
        "run_id": run_id,
        "status": "running",
        "config": cfg,
        "data_sha256": data_hash,
        "code_sha256": identity["code_sha256"],
        "provenance": [s.metadata for s in collection],
        "environment": environment_manifest(),
    }
    tick = time.perf_counter()
    with filelock.FileLock(str(folder / "collection.lock"), timeout=0):
        atomic_json(folder / "config.json", cfg)
        atomic_json(manifest_path, manifest)
        try:
            count = len(collection)
            a = max(1, int(count * cfg["split"]["train"]))
            b = max(a + 1, int(count * (cfg["split"]["train"] + cfg["split"]["validation"])))
            if b >= count:
                raise ValueError("Trajectory split leaves no test trajectories")
            inputs = (
                cfg["inputs"]
                if cfg["inputs"] is not None
                else list(range(len(collection[0].names)))
            )
            if any(s.names != collection[0].names for s in collection):
                raise ValueError("All trajectories must share observed feature names")
            targets = [inputs.index(t) for t in cfg["targets"]]
            history, horizon = cfg["history"], cfg["horizon"]
            fraction = cfg["train_fraction"]
            if not 0 < fraction <= 1:
                raise ValueError("Training fraction must be in (0,1]")
            train_raw = [
                s.observations[: int(len(s.time) * fraction), inputs] for s in collection[:a]
            ]
            if any(len(t) < history + horizon for t in train_raw):
                raise ValueError("A training trajectory is shorter than history plus horizon")
            corruption = dict(cfg["corruption"])
            protocol = corruption.pop("protocol")
            if protocol == "robust_training":
                train_raw = [
                    corrupt(t, cfg["seed"] + 1000 + i, **corruption)
                    for i, t in enumerate(train_raw)
                ]
            elif protocol != "test_time":
                raise ValueError("Unknown corruption protocol")
            scaler = Standardizer().fit(np.concatenate(train_raw))
            train = [scaler.transform(t) for t in train_raw]
            validation = [scaler.transform(s.observations[:, inputs]) for s in collection[a:b]]
            name = cfg["model"]["name"]
            dt = float(np.diff(collection[0].time)[0])
            if name in DYNAMICAL_NAMES:
                if any(
                    s.inputs is not None or not np.allclose(np.diff(s.time), dt) for s in collection
                ):
                    raise ValueError(
                        "Autonomous collection dynamics require regular, unforced trajectories"
                    )
                if (
                    name != "latent_ode"
                    and not cfg["dynamics_assumption"]
                    and any(
                        s.latent is None or len(inputs) != s.latent.shape[1] for s in collection
                    )
                ):
                    raise ValueError("Observation-space dynamics assumption must be explicit")
            if name in PRIOR_NAMES and inputs != [0, 1]:
                raise ValueError("Oscillator prior requires ordered position/velocity")
            model = make_model(name, cfg["model"], cfg["seed"])
            options = {}
            if isinstance(model, (NeuralForecaster, NeuralODE, HybridForecaster)):
                options.update(
                    deadline=deadline, checkpoint=folder / "checkpoint.pt", resume=resume
                )
                if cfg.get("tracking", {}).get("tensorboard"):
                    options["tensorboard_dir"] = folder / "tensorboard"
            if isinstance(model, (NeuralODE, HybridForecaster, MechanisticOscillator)):
                options.update(mean=scaler.mean, scale=scaler.scale)
            model.fit(train, validation, targets, history, horizon, dt, **options)
            fit_seconds = time.perf_counter() - tick
            eval_indices = (
                range(a, b) if cfg.get("evaluation", "test") == "validation" else range(b, count)
            )
            xs, truths, origins, times, trajectory_ids, histories, history_times = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for index in eval_indices:
                series = collection[index]
                raw = series.observations[:, inputs]
                altered = corrupt(raw, cfg["seed"] + 2000 + index, **corruption)
                x, _, origin = windows(
                    scaler.transform(altered), history, horizon, targets, stride=cfg["stride"]
                )
                xs.append(x)
                truths.append(np.stack([raw[o : o + horizon, targets] for o in origin]))
                origins.extend(origin.tolist())
                times.extend([series.time[o : o + horizon] for o in origin])
                histories.extend([raw[o - history : o, targets] for o in origin])
                history_times.extend([series.time[o - history : o] for o in origin])
                trajectory_ids.extend([index] * len(origin))
            truth = np.concatenate(truths)
            if not np.isfinite(truth).all():
                raise ValueError(
                    "Missing evaluation target values require explicit complete-target selection"
                )
            before = time.perf_counter()
            predicted = check_predictions(model.predict(np.concatenate(xs)), truth.shape)
            prediction = scaler.inverse(predicted, targets)
            inference = time.perf_counter() - before
            metrics = forecast_metrics(
                truth,
                prediction,
                [scaler.impute(t)[:, targets] for t in train_raw],
                cfg.get("seasonality", 1),
            )
            metrics.update(
                fit_seconds=fit_seconds,
                inference_seconds=inference,
                latency_per_origin_seconds=inference / len(truth),
                parameter_count=int(model.parameters),
                n_training_observations=sum(map(len, train_raw)),
                n_training_trajectories=a,
                n_test_origins=len(truth),
                rollout=model.rollout,
                physical_horizon=horizon * dt,
                trajectory_rmse={
                    str(i): float(
                        np.sqrt(
                            np.mean(
                                (
                                    prediction[np.array(trajectory_ids) == i]
                                    - truth[np.array(trajectory_ids) == i]
                                )
                                ** 2
                            )
                        )
                    )
                    for i in eval_indices
                },
                evaluation_split=cfg.get("evaluation", "test"),
                prediction_stability={
                    "finite_fraction": 1.0,
                    "max_absolute_value": float(np.max(np.abs(prediction))),
                },
            )
            for key in ("best_epoch", "stop_epoch", "peak_gpu_memory_bytes", "training_seconds"):
                if model.diagnostics and key in model.diagnostics:
                    metrics[key] = model.diagnostics[key]
            if cfg.get("dynamical_metrics"):
                from dynforecast.evaluation.dynamics import dynamical_metrics

                metrics["dynamical_fidelity"] = dynamical_metrics(truth, prediction, dt)
            atomic_json(folder / "metrics.json", metrics)
            atomic_json(folder / "diagnostics.json", model.diagnostics or {})
            atomic_json(
                folder / "preprocessing.json",
                {
                    "mean": scaler.mean.tolist(),
                    "scale": scaler.scale.tolist(),
                    "inputs": inputs,
                    "targets": targets,
                    "train_trajectory_ids": list(range(a)),
                    "validation_trajectory_ids": list(range(a, b)),
                    "test_trajectory_ids": list(range(b, count)),
                    "boundary_policy": "independent trajectories; windows and derivatives never cross boundaries",
                },
            )
            np.savez_compressed(
                folder / "predictions.npz",
                truth=truth,
                predictions=prediction,
                origins=np.array(origins),
                time=np.array(times)[:, 0],
                target_times=np.array(times),
                trajectory_ids=np.array(trajectory_ids),
                history=np.array(histories),
                history_times=np.array(history_times),
                target_names=np.array([collection[0].names[t] for t in cfg["targets"]]),
            )
            with (folder / "model.tmp").open("wb") as f:
                pickle.dump(model, f)
            (folder / "model.tmp").replace(folder / "model.pkl")
            if cfg.get("tracking", {}).get("mlflow"):
                from dynforecast.training.tracking import log_mlflow

                log_mlflow(folder, cfg, metrics)
            manifest.update(
                status="completed", metrics=metrics, elapsed_seconds=time.perf_counter() - tick
            )
        except BaseException as exc:
            manifest.update(
                status="interrupted"
                if isinstance(exc, (KeyboardInterrupt, TimeoutError))
                else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            (folder / "error.log").write_text(traceback.format_exc())
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            atomic_json(manifest_path, manifest)
    return manifest
