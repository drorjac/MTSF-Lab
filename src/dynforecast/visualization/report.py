"""Static experiment report: per-run figures, seed statistics, and CSV summaries.

Everything is built from saved run folders. Failed and interrupted runs are listed,
not dropped. Runs are only compared within an identical *condition* (every config
field except model and seed, plus the code and data hashes), and seeds, not
overlapping forecast windows, are the independent replicates for intervals.
"""

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

STYLE = {
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
NOT_PART_OF_CONDITION = ("model", "seed", "output", "budget_seconds", "profile")
SWEEPS = (
    ("horizon", "Forecast steps"),
    ("train_fraction", "Training observations"),
    ("noise", "Measurement noise standard deviation"),
    ("inputs", "Observed input configuration"),
)
HEADER = [
    "# DynForecastLab experiment report",
    "",
    "Recorded runs: {count}.",
    "",
    "Errors use original units. Seeds are the independent units for intervals; overlapping",
    "forecast windows are not treated as independent replicates. Three-seed development",
    "intervals are exploratory. Different datasets are never pooled on raw RMSE.",
    "",
]


class Figures:
    """Saves figures as PNG and PDF into the report folder and remembers their names."""

    def __init__(self, folder):
        self.folder, self.names = folder, []

    def save(self, fig, name):
        fig.tight_layout()
        fig.savefig(self.folder / f"{name}.png", dpi=180)
        fig.savefig(self.folder / f"{name}.pdf")
        plt.close(fig)
        self.names.append(name)


def _t_half_width(values):
    """Half-width of a 95% t interval for the mean, or None with fewer than two values."""
    n = len(values)
    if n < 2:
        return None
    return float(student_t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def _run_row(root, run):
    """One row of ``runs.csv``: config fields, status, and scalar metrics."""
    cfg = run["config"]
    condition = {k: v for k, v in cfg.items() if k not in NOT_PART_OF_CONDITION}
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
        "condition": json.dumps(condition, sort_keys=True),
        "error": run.get("error", ""),
    }
    if run["status"] == "completed":
        # Results from different code or data versions are never compared directly.
        condition["code_sha256"] = run.get("code_sha256", "legacy-unknown")
        condition["data_sha256"] = run.get("data_sha256", "legacy-unknown")
        row["condition"] = json.dumps(condition, sort_keys=True)
        metrics = json.loads((root / run["run_id"] / "metrics.json").read_text())
        row.update({k: v for k, v in metrics.items() if np.isscalar(v) or v is None})
    return row


def plot_run(root, row, figures):
    """Forecast, error-by-horizon, convergence, and phase-portrait figures for one run."""
    folder = root / row["run_id"]
    data = np.load(folder / "predictions.npz")
    metrics = json.loads((folder / "metrics.json").read_text())
    diagnostics = json.loads((folder / "diagnostics.json").read_text())

    fig, ax = plt.subplots(figsize=(7, 3))
    if "history" in data:
        ax.plot(
            data["history_times"][0],
            data["history"][0, :, 0],
            color="#a2aec0",
            label="Observed history",
        )
    target_times = data["target_times"][0]
    ax.plot(target_times, data["truth"][0, :, 0], color="#17263d", label="Observed target")
    ax.plot(target_times, data["predictions"][0, :, 0], color="#3766df", label="Forecast")
    ax.axvspan(target_times[0], target_times[-1], color="#edf3fc", zorder=-2)
    ax.set(
        xlabel="Time",
        ylabel="Original physical scale",
        title=f"{row['dataset']} · {row['model']} · first test origin",
    )
    ax.legend()
    figures.save(fig, f"forecast_{row['run_id']}")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(1, len(metrics["rmse_by_horizon"]) + 1), metrics["rmse_by_horizon"])
    ax.set(xlabel="Lead steps", ylabel="RMSE (physical scale)", title=row["model"])
    figures.save(fig, f"horizon_{row['run_id']}")

    if diagnostics.get("history"):
        history = pd.DataFrame(diagnostics["history"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for key in ("train_loss", "validation_loss"):
            axes[0].plot(history.epoch, history[key], label=key)
            axes[1].plot(history.seconds.cumsum(), history[key], label=key)
        axes[0].set(xlabel="Epoch", ylabel="Normalized MSE")
        axes[1].set(xlabel="Training seconds", ylabel="Normalized MSE")
        axes[0].legend()
        figures.save(fig, f"convergence_{row['run_id']}")

    if data["truth"].shape[-1] >= 2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(*data["truth"][0, :, :2].T, label="Truth")
        ax.plot(*data["predictions"][0, :, :2].T, label="Forecast")
        ax.set(xlabel="Target 0", ylabel="Target 1", title="Forecast phase portrait")
        ax.legend()
        figures.save(fig, f"phase_{row['run_id']}")


def label_variants(valid):
    """Name each model config: the model name, plus a short hash if it has several configs."""
    configs_per_model = valid.groupby("model").model_config.nunique()
    return [
        row.model
        if configs_per_model[row.model] == 1
        else row.model + "-" + hashlib.sha256(row.model_config.encode()).hexdigest()[:6]
        for row in valid.itertuples()
    ]


def seed_summary(valid):
    """Mean RMSE over seeds, with a 95% interval, for each condition and model config."""
    summaries = []
    for (condition, model_config), group in valid.groupby(["condition", "model_config"]):
        per_seed = group.groupby("seed").rmse.mean()
        summaries.append(
            {
                "condition": condition,
                "model_config": model_config,
                "model": group.model.iloc[0],
                "dataset": group.dataset.iloc[0],
                "seed_count": len(per_seed),
                "rmse_mean": float(per_seed.mean()),
                "ci95_half_width": _t_half_width(per_seed),
            }
        )
    return pd.DataFrame(summaries)


def pareto_front(means):
    """Variants not beaten on both fit time and RMSE by any other variant."""
    front = []
    for name, point in means.iterrows():
        no_worse = (means.fit_seconds <= point.fit_seconds) & (means.rmse <= point.rmse)
        better = (means.fit_seconds < point.fit_seconds) | (means.rmse < point.rmse)
        if not (no_worse & better).any():
            front.append(name)
    return front


def plot_cost(valid, figures):
    """Cost-vs-accuracy scatter per dataset and condition; returns the Pareto summary lines."""
    lines = []
    for dataset, group in valid.groupby("dataset"):
        for index, (_, matched) in enumerate(group.groupby("condition")):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            for variant, points in matched.groupby("variant"):
                axes[0].scatter(points.fit_seconds, points.rmse, label=variant)
                axes[1].scatter(points.parameter_count, points.rmse, label=variant)
            axes[0].set(xlabel="Fit seconds", ylabel="RMSE")
            axes[1].set(xlabel="Reported fitted parameters", ylabel="RMSE")
            axes[0].legend(fontsize=7)
            figures.save(fig, f"cost_{dataset}_{index}")
            front = pareto_front(matched.groupby("variant")[["fit_seconds", "rmse"]].mean())
            lines.append(
                f"Observed cost/accuracy Pareto models ({dataset}, condition {index}): "
                + ", ".join(front)
                + "."
            )
    return lines


def paired_comparisons(valid):
    """RMSE differences between every two variants, paired on shared seeds."""
    pairs = []
    for condition, group in valid.groupby("condition"):
        variants = sorted(group.variant.unique())
        for i, first in enumerate(variants):
            for second in variants[i + 1 :]:
                one = group[group.variant == first].groupby("seed").rmse.mean()
                two = group[group.variant == second].groupby("seed").rmse.mean()
                shared = one.index.intersection(two.index)
                if not len(shared):
                    continue
                difference = one.loc[shared] - two.loc[shared]
                pairs.append(
                    {
                        "condition": condition,
                        "first": first,
                        "second": second,
                        "paired_seeds": len(shared),
                        "rmse_difference": float(difference.mean()),
                        "ci95_half_width": _t_half_width(difference),
                    }
                )
    return pd.DataFrame(pairs)


def _without(condition, variable):
    """The condition with ``variable`` removed, so runs that differ only in it group together."""
    condition = json.loads(condition)
    if variable == "noise":
        condition["corruption"].pop("noise", None)
    else:
        condition.pop(variable, None)
    return json.dumps(condition, sort_keys=True)


def plot_sweeps(valid, figures):
    """RMSE against each swept variable, plus a model-by-horizon heatmap."""
    for variable, xlabel in SWEEPS:
        column = "n_training_observations" if variable == "train_fraction" else variable
        facets = valid.condition.map(lambda c: _without(c, variable))
        for index, (_, group) in enumerate(valid.groupby(facets)):
            if group[variable].nunique() < 2:
                continue
            fig, ax = plt.subplots(figsize=(6, 3))
            for variant, points in group.groupby("variant"):
                stats = points.groupby(column).rmse.agg(["mean", "std", "count"])
                t_values = [student_t.ppf(0.975, n - 1) if n > 1 else 0 for n in stats["count"]]
                error = stats["std"].fillna(0) / np.sqrt(stats["count"]) * np.array(t_values)
                ax.errorbar(
                    stats.index, stats["mean"], yerr=error, label=variant, marker="o", capsize=3
                )
            ax.set(
                xlabel=xlabel,
                ylabel="RMSE (physical scale)",
                title="95% seed intervals where repeated seeds exist",
            )
            ax.legend(fontsize=7)
            figures.save(fig, f"sweep_{variable}_{index}")
            if variable == "horizon":
                heatmap = group.pivot_table(index="variant", columns="horizon", values="rmse")
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(heatmap.values, aspect="auto")
                ax.set_xticks(range(len(heatmap.columns)), heatmap.columns)
                ax.set_yticks(range(len(heatmap.index)), heatmap.index)
                ax.set(xlabel="Forecast horizon", ylabel="Model")
                fig.colorbar(im, ax=ax, label="RMSE")
                figures.save(fig, f"heatmap_{index}")


def write_html(report, header, frame, figures):
    columns = ("run_id", "dataset", "model", "status", "rmse", "mae", "fit_seconds", "error")
    visible = [c for c in columns if c in frame]
    table = (
        frame[visible].to_html(index=False, escape=True)
        if len(frame)
        else "<p>No recorded runs.</p>"
    )
    images = "".join(
        f'<figure><img src="{f}.png" width="800"><figcaption>{html.escape(f)}</figcaption></figure>'
        for f in figures.names
    )
    (report / "summary.html").write_text(
        "<!doctype html><html><meta charset='utf-8'><title>DynForecastLab</title><body>"
        "<h1>DynForecastLab experiment report</h1>"
        f"<pre>{html.escape(chr(10).join(header))}</pre>{table}{images}</body></html>"
    )


def generate_report(root="results"):
    """Build ``<root>/report`` (Markdown, HTML, CSVs, figures) from the runs in ``root``."""
    plt.rcParams.update(STYLE)
    root = Path(root)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    figures = Figures(report)

    manifests = [json.loads(p.read_text()) for p in sorted(root.glob("*/manifest.json"))]
    rows = []
    for run in manifests:
        row = _run_row(root, run)
        if run["status"] == "completed":
            plot_run(root, row, figures)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(report / "runs.csv", index=False)

    header = [line.format(count=len(rows)) for line in HEADER]
    text = list(header)
    model_configs = {run["run_id"]: run["config"]["model"] for run in manifests}
    valid = pd.DataFrame([r for r in rows if r["status"] == "completed"])
    if len(valid):
        # Different settings of the same model are compared as separate variants.
        valid["model_config"] = [
            json.dumps(model_configs[run_id], sort_keys=True) for run_id in valid.run_id
        ]
        valid["variant"] = label_variants(valid)
        seed_summary(valid).to_csv(report / "seed_summary.csv", index=False)
        columns = ["dataset", "model", "horizon", "seed", "rmse", "mae", "mase", "fit_seconds"]
        text += ["```", valid[columns].to_string(index=False), "```", ""]
        text += plot_cost(valid, figures)
        paired_comparisons(valid).to_csv(report / "paired_comparisons.csv", index=False)
        plot_sweeps(valid, figures)
        equations = valid[valid.model == "sindy"]
        if len(equations):
            equations.to_csv(report / "equation_recovery.csv", index=False)
    if len(frame):
        failures = frame[frame.status != "completed"]
        if len(failures):
            text += [
                "## Failed or interrupted runs",
                "",
                "```",
                failures[["run_id", "model", "status", "error"]].to_string(index=False),
                "```",
            ]
    text += ["", "## Figures", ""] + [f"![{f}]({f}.png)" for f in figures.names]
    (report / "summary.md").write_text("\n".join(text))
    write_html(report, header, frame, figures)

    from .tables import research_tables

    research_tables(root, report)
    try:
        from .dashboard import generate_dashboard

        generate_dashboard(root)
    except ImportError:
        pass  # the static report works without the optional plotly install
    return report
