from pathlib import Path
import json


def log_mlflow(folder, config, metrics):
    import mlflow

    folder = Path(folder).resolve()
    tracking = Path(config.get("tracking", {}).get("directory", folder.parent / "mlruns")).resolve()
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking.as_uri())
    mlflow.set_experiment("DynForecastLab")
    with mlflow.start_run(run_name=folder.name):
        mlflow.log_params(
            {
                "model": config["model"]["name"],
                "dataset": config["dataset"]["name"],
                "seed": config["seed"],
                "horizon": config["horizon"],
            }
        )
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v is not None
            }
        )
        mlflow.log_artifacts(str(folder), artifact_path="experiment")
        (folder / "mlflow.json").write_text(
            json.dumps(
                {"run_id": mlflow.active_run().info.run_id, "tracking_directory": str(tracking)},
                indent=2,
            )
        )
