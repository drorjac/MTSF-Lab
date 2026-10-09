"""Matched corruption degradation and explicitly tagged cross-dataset summaries."""

import json
from pathlib import Path
import pandas as pd


def research_tables(root, report):
    root, report = Path(root), Path(report)
    runs = [json.loads(p.read_text()) for p in sorted(root.glob("*/manifest.json"))]
    completed = [r for r in runs if r["status"] == "completed"]
    rows = []

    def matched_key(run):
        cfg = dict(run["config"])
        for field in ("output", "profile", "budget_seconds", "tracking", "scheduler"):
            cfg.pop(field, None)
        corruption = cfg.pop("corruption")
        cfg["protocol"] = corruption["protocol"]
        cfg["code_sha256"] = run.get("code_sha256")
        cfg["data_sha256"] = run.get("data_sha256")
        return json.dumps(cfg, sort_keys=True)

    clean = {}
    for run in completed:
        corruption = run["config"]["corruption"]
        if all(
            not corruption.get(key, 0) for key in ("noise", "missing", "outliers", "bias", "outage")
        ):
            clean[matched_key(run)] = run
    for run in completed:
        reference = clean.get(matched_key(run))
        rmse = run["metrics"]["rmse"]
        base = None if reference is None else reference["metrics"]["rmse"]
        rows.append(
            {
                "run_id": run["run_id"],
                "model": run["config"]["model"]["name"],
                "protocol": run["config"]["corruption"]["protocol"],
                "corruption": json.dumps(run["config"]["corruption"]),
                "rmse": rmse,
                "clean_rmse": base,
                "rmse_increase": None if base is None else rmse - base,
                "degradation_ratio": None if base is None or base < 1e-12 else rmse / base,
                "clean_run_id": None if reference is None else reference["run_id"],
            }
        )
    pd.DataFrame(rows).to_csv(report / "robustness_degradation.csv", index=False)
    tagged = []
    for run in completed:
        tag = run["config"].get("comparison_tag")
        if tag:
            tagged.append(
                {
                    "tag": tag,
                    "dataset": run["config"]["dataset"]["name"],
                    "model": run["config"]["model"]["name"],
                    "mase": run["metrics"]["mase"],
                    "seed": run["config"]["seed"],
                }
            )
    if tagged:
        frame = pd.DataFrame(tagged)
        frame.to_csv(report / "tagged_dataset_scores.csv", index=False)
        summaries = []
        for tag, group in frame.groupby("tag"):
            scores = group.groupby(["dataset", "model"]).mase.mean().unstack("model").dropna(axis=1)
            for model in scores.columns:
                summaries.append(
                    {
                        "tag": tag,
                        "model": model,
                        "dataset_count": len(scores),
                        "macro_average_mase": float(scores[model].mean()),
                        "aggregation": "equal dataset weighting; common model coverage only; tagged tasks",
                    }
                )
        pd.DataFrame(summaries).to_csv(report / "cross_dataset_mase.csv", index=False)
    failures = {}
    for run in runs:
        name = run["config"]["model"]["name"]
        record = failures.setdefault(
            name, {"model": name, "attempted": 0, "failed": 0, "interrupted": 0}
        )
        record["attempted"] += 1
        record["failed"] += run["status"] == "failed"
        record["interrupted"] += run["status"] == "interrupted"
    for record in failures.values():
        record["failure_rate"] = record["failed"] / record["attempted"]
    pd.DataFrame(list(failures.values())).to_csv(report / "failure_rates.csv", index=False)
