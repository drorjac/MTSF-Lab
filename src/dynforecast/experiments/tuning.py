"""Validation-only, bounded Optuna search followed by one held-out evaluation."""

import copy
import hashlib
import json
from pathlib import Path
import time
from .runner import DEFAULT, merge, run_experiment, set_nested, atomic_json, code_fingerprint


def tune(configuration, resume=True):
    import optuna

    cfg = merge(DEFAULT, configuration)
    search = cfg.pop("tuning", {})
    space = search.get("space", {"model.width": [8, 16, 32], "model.lr": [0.001, 0.005]})
    if any(not key.startswith("model.") for key in space):
        raise ValueError(
            "Tuning may change model settings only; data, targets and horizons remain fixed"
        )
    if any(not isinstance(v, list) or not v for v in space.values()):
        raise ValueError("Tuning choices must be nonempty lists")
    trials = search.get("trials", 10)
    seconds = search.get("budget_seconds", 600)
    if trials < 1 or seconds <= 0:
        raise ValueError("Tuning trials and budget must be positive")
    root = Path(cfg["output"]) / "tuning"
    root.mkdir(parents=True, exist_ok=True)
    identity = json.dumps(
        {"config": cfg, "space": space, "source": code_fingerprint()}, sort_keys=True
    )
    name = hashlib.sha256(identity.encode()).hexdigest()[:16]
    study = optuna.create_study(
        study_name=name,
        storage="sqlite:///" + str((root / "study.db").resolve()),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        load_if_exists=resume,
    )
    deadline = time.monotonic() + seconds

    def objective(trial):
        candidate = copy.deepcopy(cfg)
        for key, options in space.items():
            set_nested(candidate, key, trial.suggest_categorical(key, options))
        candidate.update(evaluation="validation", output=str(root / "trials"))
        outcome = run_experiment(candidate, resume, deadline)
        trial.set_user_attr("run_id", outcome["run_id"])
        trial.set_user_attr("status", outcome["status"])
        if outcome["status"] != "completed":
            trial.set_user_attr("error", outcome.get("error"))
            raise optuna.TrialPruned(outcome.get("error", "incomplete experiment"))
        return outcome["metrics"]["rmse"]

    finished = sum(t.state.is_finished() for t in study.trials)
    if finished < trials:
        study.optimize(objective, n_trials=trials - finished, timeout=seconds)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        result = {"status": "failed", "error": "No successful validation trials", "study": name}
        atomic_json(root / "selection.json", result)
        return result
    best = copy.deepcopy(cfg)
    for key, value in study.best_params.items():
        set_nested(best, key, value)
    best.update(evaluation="test", output=str(root / "selected"))
    # Held-out test is evaluated once after selection; never read by the optimizer.
    final = run_experiment(best, resume)
    result = {
        "status": final["status"],
        "study": name,
        "selection_metric": "validation RMSE",
        "best_parameters": study.best_params,
        "best_validation_rmse": study.best_value,
        "completed_trials": len(completed),
        "test_run_id": final["run_id"],
        "test_metrics": final.get("metrics"),
        "predeclared_trial_budget": trials,
    }
    atomic_json(root / "selection.json", result)
    return result


def backtest(configuration):
    cfg = merge(DEFAULT, configuration)
    folds = cfg.pop("backtest", {}).get("folds", [0.3, 0.45, 0.6])
    results = []
    for index, train_fraction in enumerate(folds):
        candidate = copy.deepcopy(cfg)
        candidate.update(
            split={"train": train_fraction, "validation": 0.15}, evaluation="validation"
        )
        candidate["output"] = str(Path(cfg["output"]) / "backtests")
        outcome = run_experiment(candidate)
        results.append(
            {
                "fold": index,
                "train_fraction": train_fraction,
                "status": outcome["status"],
                "run_id": outcome["run_id"],
                "validation_rmse": outcome.get("metrics", {}).get("rmse"),
                "error": outcome.get("error"),
            }
        )
    summary = {
        "status": "completed" if all(r["status"] == "completed" for r in results) else "failed",
        "protocol": "expanding chronological training prefixes; split-local validation targets",
        "folds": results,
    }
    root = Path(cfg["output"]) / "backtests"
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "summary.json", summary)
    return summary
