import json
from dynforecast.experiments import run_benchmark
from dynforecast.visualization import generate_report


def test_report_horizon_sweep_intervals_and_heatmap(tmp_path):
    cfg = {
        "dataset": {"n": 160},
        "history": 8,
        "targets": [0],
        "output": str(tmp_path),
        "budget_seconds": 60,
        "sweep": {"model.name": ["persistence", "var"], "horizon": [2, 4], "seed": [0, 1]},
    }
    outcome = run_benchmark(cfg)
    assert outcome["completed"] == 8
    folder = generate_report(tmp_path)
    assert (folder / "heatmap_0.png").exists()
    assert (folder / "sweep_horizon_0.pdf").exists()
    import pandas as pd

    summary = pd.read_csv(folder / "seed_summary.csv")
    assert (summary.seed_count == 2).all()
    assert summary.ci95_half_width.notna().all()
    paired = pd.read_csv(folder / "paired_comparisons.csv")
    assert (paired.paired_seeds == 2).all()
    assert json.loads((tmp_path / "benchmark.json").read_text())["failed"] == 0
