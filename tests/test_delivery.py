import io
import json
import zipfile
import pytest
import pandas as pd
from dynforecast.data.collections import download_tsf
from dynforecast.experiments.runner import run_experiment, run_benchmark
from dynforecast.visualization.tables import research_tables
from dynforecast.cli import parse_config


def tsf(offset=0):
    return (
        "@attribute series_name string\n@data\n"
        + "\n".join(
            f"s{i}:" + ",".join(str(offset + i + j / 10) for j in range(100)) for i in range(5)
        )
    ).encode()


def test_tsf_members_have_distinct_validated_caches(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a.tsf", tsf())
        z.writestr("b.tsf", tsf(20))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(archive.getvalue()))
    with pytest.raises(ValueError, match="Select a TSF member"):
        download_tsf("https://example.com/data.zip", tmp_path)
    a = download_tsf("https://example.com/data.zip", tmp_path, "a.tsf")
    b = download_tsf("https://example.com/data.zip", tmp_path, "b.tsf")
    assert a != b and a.read_bytes() != b.read_bytes()
    result = run_experiment(
        {
            "dataset": {"name": "monash", "path": str(a)},
            "model": {"name": "ridge"},
            "history": 8,
            "horizon": 3,
            "targets": [0],
            "output": str(tmp_path / "runs"),
        }
    )
    assert result["status"] == "completed", result.get("error")
    assert result["metrics"]["n_training_trajectories"] == 3


def test_invalid_tsf_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"not a dataset"))
    with pytest.raises(ValueError, match="No TSF series"):
        download_tsf("https://example.com/invalid.tsf", tmp_path)
    assert not list(tmp_path.glob("*.tsf"))


def test_matching_degradation_and_macro_scores(tmp_path):
    root, report = tmp_path / "runs", tmp_path / "report"
    report.mkdir()
    configs = [
        {"dataset": {"name": "oscillator", "n": 160}, "corruption": {"noise": noise}}
        for noise in [0, 0.1]
    ] + [
        {"dataset": {"name": name, "max_rows": 150}, "comparison_tag": "measurements"}
        for name in ["sunspots", "macrodata"]
    ]
    for cfg in configs:
        result = run_experiment({**cfg, "history": 8, "horizon": 3, "output": str(root)})
        assert result["status"] == "completed"
    research_tables(root, report)
    degradation = pd.read_csv(report / "robustness_degradation.csv")
    noisy = degradation[degradation.corruption.str.contains("0.1")].iloc[0]
    assert noisy.clean_run_id and noisy.degradation_ratio > 0
    scores = pd.read_csv(report / "cross_dataset_mase.csv")
    assert scores.iloc[0].dataset_count == 2
    assert scores.iloc[0].macro_average_mase > 0


def test_isolated_scheduler_completes_and_auto_device_resolves(tmp_path):
    result = run_benchmark(
        {
            "dataset": {"n": 160},
            "history": 8,
            "horizon": 3,
            "output": str(tmp_path),
            "budget_seconds": 60,
            "scheduler": {"isolated": True, "cpu_workers": 1},
            "model": {"device": "auto"},
            "sweep": {"seed": [0]},
        }
    )
    assert result["completed"] == 1 and result["failed"] == 0
    record = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert record["config"]["model"]["device"] != "auto"


def test_config_composition_and_numeric_interpolation(tmp_path):
    (tmp_path / "base.yaml").write_text("history: 8\nmodel: {name: ridge}\n")
    (tmp_path / "main.yaml").write_text("includes: [base.yaml]\nhorizon: ${history}\n")
    cfg = parse_config(tmp_path / "main.yaml", ["history=12"])
    assert cfg["horizon"] == 12 and cfg["model"]["name"] == "ridge"
    with pytest.raises(ValueError, match="credential interpolation"):
        parse_config(None, ["model.secret=${oc.env:FAKE_SECRET}"])
