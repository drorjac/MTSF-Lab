"""Figures contain only saved completed experiment results. Failures remain visible."""

from pathlib import Path
import html
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=180)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)


def generate_report(root="results"):
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e9edf4",
            "grid.linewidth": 0.7,
            "axes.labelcolor": "#47566d",
            "text.color": "#17263d",
            "font.family": "DejaVu Sans",
            "axes.prop_cycle": plt.cycler(color=["#3766df", "#159b91", "#ef795d", "#9672cf"]),
        }
    )
    root = Path(root)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads(p.read_text()) for p in sorted(root.glob("*/manifest.json"))]
    rows, figures = [], []
    for run in manifests:
        cfg = run["config"]
        row = {
            "run_id": run["run_id"],
            "status": run["status"],
            "model": cfg["model"]["name"],
            "dataset": cfg["dataset"]["name"],
            "horizon": cfg["horizon"],
            "seed": cfg["seed"],
            "train_fraction": cfg["train_fraction"],
            "inputs": str(cfg["inputs"]),
            "targets": str(cfg["targets"]),
            "protocol": cfg["corruption"]["protocol"],
            "noise": cfg["corruption"].get("noise", 0.0),
            "missing": cfg["corruption"].get("missing", 0.0),
            "condition": json.dumps(
                {
                    k: v
                    for k, v in cfg.items()
                    if k not in ("model", "seed", "output", "budget_seconds", "profile")
                },
                sort_keys=True,
            ),
            "error": run.get("error", ""),
        }
        if run["status"] == "completed":
            condition = json.loads(row["condition"])
            condition["code_sha256"] = run.get("code_sha256", "legacy-unknown")
            condition["data_sha256"] = run.get("data_sha256", "legacy-unknown")
            row["condition"] = json.dumps(condition, sort_keys=True)
            metrics = json.loads((root / run["run_id"] / "metrics.json").read_text())
            row.update({k: v for k, v in metrics.items() if np.isscalar(v) or v is None})
            data = np.load(root / run["run_id"] / "predictions.npz")
            fig, ax = plt.subplots(figsize=(7, 3))
            if "history" in data:
                ax.plot(
                    data["history_times"][0],
                    data["history"][0, :, 0],
                    color="#a2aec0",
                    label="Observed history",
                )
            ax.plot(
                data["target_times"][0],
                data["truth"][0, :, 0],
                color="#17263d",
                label="Observed target",
            )
            ax.plot(
                data["target_times"][0],
                data["predictions"][0, :, 0],
                color="#3766df",
                label="Forecast",
            )
            ax.axvspan(
                data["target_times"][0, 0], data["target_times"][0, -1], color="#edf3fc", zorder=-2
            )
            ax.set(
                xlabel="Time",
                ylabel="Original physical scale",
                title=f"{row['dataset']} · {row['model']} · first test origin",
            )
            ax.legend()
            name = f"forecast_{run['run_id']}"
            _save(fig, report, name)
            figures.append(name)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(np.arange(1, len(metrics["rmse_by_horizon"]) + 1), metrics["rmse_by_horizon"])
            ax.set(xlabel="Lead steps", ylabel="RMSE (physical scale)", title=row["model"])
            name = f"horizon_{run['run_id']}"
            _save(fig, report, name)
            figures.append(name)
            diagnostics = json.loads((root / run["run_id"] / "diagnostics.json").read_text())
            if diagnostics.get("history"):
                history = pd.DataFrame(diagnostics["history"])
                fig, axes = plt.subplots(1, 2, figsize=(9, 3))
                for key in ("train_loss", "validation_loss"):
                    axes[0].plot(history.epoch, history[key], label=key)
                    axes[1].plot(history.seconds.cumsum(), history[key], label=key)
                axes[0].set(xlabel="Epoch", ylabel="Normalized MSE")
                axes[1].set(xlabel="Training seconds", ylabel="Normalized MSE")
                axes[0].legend()
                name = f"convergence_{run['run_id']}"
                _save(fig, report, name)
                figures.append(name)
            if data["truth"].shape[-1] >= 2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.plot(*data["truth"][0, :, :2].T, label="Truth")
                ax.plot(*data["predictions"][0, :, :2].T, label="Forecast")
                ax.set(xlabel="Target 0", ylabel="Target 1", title="Forecast phase portrait")
                ax.legend()
                name = f"phase_{run['run_id']}"
                _save(fig, report, name)
                figures.append(name)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(report / "runs.csv", index=False)
    text = [
        "# DynForecastLab experiment report",
        "",
        f"Recorded runs: {len(rows)}.",
        "",
        "Errors use original units. Seeds are the independent units for intervals; overlapping",
        "forecast windows are not treated as independent replicates. Three-seed development",
        "intervals are exploratory. Different datasets are never pooled on raw RMSE.",
        "",
    ]
    summaries = []
    if len(frame):
        valid = frame[frame.status == "completed"].copy()
        # Exact condition/model config matching prevents pooling tuning or corruption variants.
        for row in rows:
            row["model_config"] = json.dumps(
                next(r["config"]["model"] for r in manifests if r["run_id"] == row["run_id"]),
                sort_keys=True,
            )
        valid = pd.DataFrame([r for r in rows if r["status"] == "completed"])
        if len(valid):
            counts = valid.groupby("model").model_config.nunique()
            valid["variant"] = [
                row.model
                if counts[row.model] == 1
                else row.model + "-" + hashlib.sha256(row.model_config.encode()).hexdigest()[:6]
                for row in valid.itertuples()
            ]
            for (condition, model_cfg), group in valid.groupby(["condition", "model_config"]):
                independent = group.groupby("seed").rmse.mean()
                mean = float(independent.mean())
                interval = None
                if len(independent) >= 2:
                    interval = float(
                        student_t.ppf(0.975, len(independent) - 1)
                        * independent.std(ddof=1)
                        / np.sqrt(len(independent))
                    )
                summaries.append(
                    {
                        "condition": condition,
                        "model_config": model_cfg,
                        "model": group.model.iloc[0],
                        "dataset": group.dataset.iloc[0],
                        "seed_count": len(independent),
                        "rmse_mean": mean,
                        "ci95_half_width": interval,
                    }
                )
            pd.DataFrame(summaries).to_csv(report / "seed_summary.csv", index=False)
            columns = ["dataset", "model", "horizon", "seed", "rmse", "mae", "mase", "fit_seconds"]
            text.extend(["```", valid[columns].to_string(index=False), "```", ""])
            for dataset, group in valid.groupby("dataset"):
                # A facet is an exact experimental condition, preventing unfair mixed-task comparisons.
                for index, (_, matched) in enumerate(group.groupby("condition")):
                    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
                    for model_name, points in matched.groupby("variant"):
                        axes[0].scatter(points.fit_seconds, points.rmse, label=model_name)
                        axes[1].scatter(points.parameter_count, points.rmse, label=model_name)
                    axes[0].set(xlabel="Fit seconds", ylabel="RMSE")
                    axes[1].set(xlabel="Reported fitted parameters", ylabel="RMSE")
                    axes[0].legend(fontsize=7)
                    name = f"cost_{dataset}_{index}"
                    _save(fig, report, name)
                    figures.append(name)
                    means = matched.groupby("variant")[["fit_seconds", "rmse"]].mean()
                    pareto = []
                    for model_name, point in means.iterrows():
                        dominated = (
                            (means.fit_seconds <= point.fit_seconds)
                            & (means.rmse <= point.rmse)
                            & ((means.fit_seconds < point.fit_seconds) | (means.rmse < point.rmse))
                        ).any()
                        if not dominated:
                            pareto.append(model_name)
                    text.append(
                        f"Observed cost/accuracy Pareto models ({dataset}, condition {index}): "
                        + ", ".join(pareto)
                        + "."
                    )
            # Pair seeds and identical condition; never infer independent samples from overlapping origins.
            pairs = []
            for condition, group in valid.groupby("condition"):
                models = sorted(group.variant.unique())
                for i, first in enumerate(models):
                    for second in models[i + 1 :]:
                        one = group[group.variant == first].groupby("seed").rmse.mean()
                        two = group[group.variant == second].groupby("seed").rmse.mean()
                        shared = one.index.intersection(two.index)
                        if not len(shared):
                            continue
                        difference = one.loc[shared] - two.loc[shared]
                        half = (
                            None
                            if len(shared) < 2
                            else float(
                                student_t.ppf(0.975, len(shared) - 1)
                                * difference.std(ddof=1)
                                / np.sqrt(len(shared))
                            )
                        )
                        pairs.append(
                            {
                                "condition": condition,
                                "first": first,
                                "second": second,
                                "paired_seeds": len(shared),
                                "rmse_difference": float(difference.mean()),
                                "ci95_half_width": half,
                            }
                        )
            pd.DataFrame(pairs).to_csv(report / "paired_comparisons.csv", index=False)
            for variable, xlabel in (
                ("horizon", "Forecast steps"),
                ("train_fraction", "Training observations"),
                ("noise", "Measurement noise standard deviation"),
                ("inputs", "Observed input configuration"),
            ):
                faceted = valid.copy()

                def facet(condition):
                    condition = json.loads(condition)
                    if variable == "noise":
                        condition["corruption"].pop("noise", None)
                    else:
                        condition.pop(variable, None)
                    return json.dumps(condition, sort_keys=True)

                faceted["facet"] = faceted.condition.map(facet)
                for index, (_, group) in enumerate(faceted.groupby("facet")):
                    if group[variable].nunique() < 2:
                        continue
                    fig, ax = plt.subplots(figsize=(6, 3))
                    for model_name, points in group.groupby("variant"):
                        column = (
                            "n_training_observations" if variable == "train_fraction" else variable
                        )
                        aggregate = points.groupby(column).rmse.agg(["mean", "std", "count"])
                        error = aggregate["std"].fillna(0) / np.sqrt(aggregate["count"])
                        error *= np.array(
                            [
                                student_t.ppf(0.975, n - 1) if n > 1 else 0
                                for n in aggregate["count"]
                            ]
                        )
                        ax.errorbar(
                            aggregate.index,
                            aggregate["mean"],
                            yerr=error,
                            label=model_name,
                            marker="o",
                            capsize=3,
                        )
                    ax.set(
                        xlabel=xlabel,
                        ylabel="RMSE (physical scale)",
                        title="95% seed intervals where repeated seeds exist",
                    )
                    ax.legend(fontsize=7)
                    name = f"sweep_{variable}_{index}"
                    _save(fig, report, name)
                    figures.append(name)
                    if variable == "horizon":
                        heatmap = group.pivot_table(
                            index="variant", columns="horizon", values="rmse"
                        )
                        fig, ax = plt.subplots(figsize=(6, 4))
                        im = ax.imshow(heatmap.values, aspect="auto")
                        ax.set_xticks(range(len(heatmap.columns)), heatmap.columns)
                        ax.set_yticks(range(len(heatmap.index)), heatmap.index)
                        ax.set(xlabel="Forecast horizon", ylabel="Model")
                        fig.colorbar(im, ax=ax, label="RMSE")
                        name = f"heatmap_{index}"
                        _save(fig, report, name)
                        figures.append(name)
            equations = valid[valid.model == "sindy"]
            if len(equations):
                equations.to_csv(report / "equation_recovery.csv", index=False)
        failures = frame[frame.status != "completed"]
        if len(failures):
            text.extend(
                [
                    "## Failed or interrupted runs",
                    "",
                    "```",
                    failures[["run_id", "model", "status", "error"]].to_string(index=False),
                    "```",
                ]
            )
    text.extend(["", "## Figures", ""] + [f"![{f}]({f}.png)" for f in figures])
    (report / "summary.md").write_text("\n".join(text))
    body = (
        "<h1>DynForecastLab experiment report</h1><pre>"
        + html.escape("\n".join(text[:9]))
        + "</pre>"
    )
    visible = [
        c
        for c in ("run_id", "dataset", "model", "status", "rmse", "mae", "fit_seconds", "error")
        if c in frame
    ]
    body += (
        frame[visible].to_html(index=False, escape=True)
        if len(frame)
        else "<p>No recorded runs.</p>"
    )
    body += "".join(
        f'<figure><img src="{f}.png" width="800"><figcaption>{html.escape(f)}</figcaption></figure>'
        for f in figures
    )
    (report / "summary.html").write_text(
        "<!doctype html><html><meta charset='utf-8'><title>DynForecastLab</title>"
        "<body>" + body + "</body></html>"
    )
    from .tables import research_tables

    research_tables(root, report)
    try:
        from .dashboard import generate_dashboard

        generate_dashboard(root)
    except ImportError:
        pass  # Core-only installations still produce the static report.
    return report
