import argparse
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from dynforecast.experiments.runner import DEFAULT, merge, run_experiment, run_benchmark, set_nested
from dynforecast.data import download_dataset
from dynforecast.simulations import simulate
from dynforecast.visualization import generate_report


def parse_config(path, overrides):
    def read_config(path):
        data = yaml.safe_load(Path(path).read_text()) or {}
        base = {}
        for include in data.pop("includes", []):
            base = merge(base, read_config(Path(path).parent / include))
        return merge(base, data)

    configuration = {} if path is None else read_config(path)
    cfg = merge(DEFAULT, configuration or {})
    for argument in overrides:
        if "=" not in argument:
            raise ValueError(f"Expected key=value override, got {argument}")
        key, value = argument.split("=", 1)
        if key in ("model", "dataset"):
            key += ".name"
        set_nested(cfg, key, yaml.safe_load(value))
    if "${" in json.dumps(cfg):
        if "${oc.env" in json.dumps(cfg):
            raise ValueError(
                "Environment/credential interpolation is not stored in experiment configs"
            )
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DynForecastLab experiments and scientific reports"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "benchmark", "tune", "backtest"):
        p = subs.add_parser(command)
        p.add_argument("--config")
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("overrides", nargs="*")
    p = subs.add_parser("simulate")
    p.add_argument("--system", default="lorenz")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--trajectories", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--output", default="data/simulations")
    p = subs.add_parser("download")
    p.add_argument("--dataset", choices=["ett", "jena", "weather"], required=True)
    p.add_argument("--output", default="data/raw")
    p = subs.add_parser("report")
    p.add_argument("--results", default="results")
    p = subs.add_parser("dashboard")
    p.add_argument("--results", default="results")
    args = parser.parse_args(argv)
    if args.command == "simulate":
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        if args.trajectories < 1:
            parser.error("trajectories must be positive")
        rng = np.random.default_rng(args.seed)
        for index in range(args.trajectories):
            initial = None
            if index:
                from dynforecast.simulations import SYSTEMS

                if args.system in SYSTEMS:
                    initial = np.asarray(SYSTEMS[args.system]["initial"]) * (
                        1 + rng.normal(0, 0.15, len(SYSTEMS[args.system]["initial"]))
                    )
            data = simulate(
                args.system, args.n, args.dt, args.seed + index, args.noise, initial=initial
            )
            values = {"time": data.time, "observations": data.observations, "latent": data.latent}
            if data.derivatives is not None:
                values["derivatives"] = data.derivatives
            if data.inputs is not None:
                values["inputs"] = data.inputs
            stem = path / f"{args.system}_{index}"
            np.savez_compressed(stem.with_suffix(".npz"), **values)
            stem.with_suffix(".json").write_text(json.dumps(data.metadata, indent=2))
        print(path)
    elif args.command == "download":
        print(download_dataset(args.dataset, args.output))
    elif args.command == "dashboard":
        from dynforecast.visualization.dashboard import generate_dashboard

        print(generate_dashboard(args.results))
    elif args.command == "report":
        print(generate_report(args.results))
    else:
        cfg = parse_config(args.config, args.overrides)
        if args.command == "tune":
            from dynforecast.experiments.tuning import tune

            outcome = tune(cfg, not args.no_resume)
        elif args.command == "backtest":
            from dynforecast.experiments.tuning import backtest

            outcome = backtest(cfg)
        else:
            outcome = (
                run_experiment(cfg, not args.no_resume)
                if args.command == "run"
                else run_benchmark(cfg, not args.no_resume)
            )
        print(json.dumps(outcome, indent=2))
        if (
            outcome.get("status") in ("failed", "interrupted")
            or outcome.get("failed", 0)
            or outcome.get("interrupted", 0)
        ):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
