"""Portable, offline interactive scientific dashboard over actual recorded artifacts."""

from pathlib import Path
import json
import numpy as np


def generate_dashboard(root="results"):
    from plotly.offline import get_plotlyjs

    root = Path(root)
    runs, failures = [], []
    for path in sorted(root.rglob("manifest.json")):
        if "mlruns" in path.parts:
            continue
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            failures.append(
                {
                    "id": record["run_id"],
                    "status": record["status"],
                    "model": record["config"]["model"]["name"],
                    "error": record.get("error", ""),
                }
            )
            continue
        data_path = path.parent / "predictions.npz"
        if not data_path.exists():
            continue
        with np.load(data_path, allow_pickle=False) as arrays:
            samples = np.linspace(
                0, len(arrays["truth"]) - 1, min(150, len(arrays["truth"])), dtype=int
            )
            data = {
                k: arrays[k][samples].tolist() for k in ("truth", "predictions", "target_times")
            }
            for key in ("history", "history_times", "trajectory_ids"):
                if key in arrays:
                    data[key] = arrays[key][samples].tolist()
            data["target_names"] = (
                arrays["target_names"].tolist()
                if "target_names" in arrays
                else [f"Target {i}" for i in range(arrays["truth"].shape[-1])]
            )
            data["sample_indices"] = samples.tolist()
        config = record["config"]
        condition = {
            k: v
            for k, v in config.items()
            if k
            not in (
                "seed",
                "model",
                "output",
                "budget_seconds",
                "scheduler",
                "tracking",
                "profile",
            )
        }
        condition.pop("horizon", None)
        condition["evaluation"] = config.get("evaluation", "test")
        condition["code_sha256"] = record.get("code_sha256")
        condition["data_sha256"] = record.get("data_sha256")
        runs.append(
            {
                "id": record["run_id"],
                "dataset": config["dataset"]["name"],
                "model": config["model"]["name"],
                "config": config,
                "metrics": json.loads((path.parent / "metrics.json").read_text()),
                "diagnostics": json.loads((path.parent / "diagnostics.json").read_text()),
                "data": data,
                "condition": json.dumps(condition, sort_keys=True),
                "time_origin": record.get("provenance", {}).get("time_origin")
                if isinstance(record.get("provenance"), dict)
                else None,
                "time_units": record.get("provenance", {}).get("time_units", "dataset units")
                if isinstance(record.get("provenance"), dict)
                else "dataset units",
            }
        )
    payload = json.dumps({"runs": runs, "failures": failures}, allow_nan=False).replace(
        "<", "\\u003c"
    )
    folder = root / "dashboard"
    folder.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("dashboard.html").read_text()
    target = folder / "index.html"
    target.write_text(
        template.replace("__PLOTLY_LIBRARY__", get_plotlyjs()).replace(
            "__EXPERIMENT_DATA__", payload
        )
    )
    return target
