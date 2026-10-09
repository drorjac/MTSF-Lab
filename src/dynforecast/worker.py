"""A killable experiment process for hard-budget scheduling."""

import argparse
import json
from pathlib import Path
import signal
from dynforecast.experiments.runner import run_experiment, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("result")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    def terminated(signum, frame):
        raise TimeoutError("Scheduler terminated this job at its wall-clock budget")

    signal.signal(signal.SIGTERM, terminated)
    result = run_experiment(json.loads(Path(args.config).read_text()), not args.no_resume)
    atomic_json(args.result, result)


if __name__ == "__main__":
    main()
