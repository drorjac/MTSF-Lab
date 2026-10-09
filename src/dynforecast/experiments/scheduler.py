import copy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from .runner import DEFAULT, merge, set_nested, atomic_json


def isolated_run(config, seconds, resume=True):
    root = Path(config["output"]) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    task = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    cfg_path, result_path, log_path = (
        root / f"{task}.config.json",
        root / f"{task}.result.json",
        root / f"{task}.log",
    )
    atomic_json(cfg_path, config)
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "dynforecast.worker",
        str(cfg_path.resolve()),
        str(result_path.resolve()),
    ]
    if not resume:
        command.append("--no-resume")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix"
        )
        interrupted = False
        try:
            process.wait(timeout=max(0.001, seconds))
        except subprocess.TimeoutExpired:
            interrupted = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
    if result_path.exists():
        result = json.loads(result_path.read_text())
        result["worker_log"] = str(log_path)
        return result
    result = {
        "run_id": task,
        "status": "interrupted" if interrupted else "failed",
        "config": config,
        "error": "Hard budget terminated job before a result was returned"
        if interrupted
        else f"Worker exited with status {process.returncode}; inspect {log_path}",
        "worker_log": str(log_path),
    }
    # Discovery/reporting includes jobs that were killed before creating their experiment manifest.
    folder = Path(config["output"]) / f"worker-{task}"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "manifest.json", result)
    return result


def scheduled_benchmark(configuration, resume=True):
    cfg = merge(DEFAULT, configuration)
    settings = cfg.get("scheduler", {})
    devices = settings.get("devices")
    if devices is None:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    if not devices or any(d.startswith("cuda") for d in devices) and not torch.cuda.is_available():
        raise ValueError("Scheduler devices must be available on this machine")
    cpu_workers = settings.get("cpu_workers", 1)
    if devices == ["cpu"]:
        devices *= cpu_workers
    sweep = cfg.get("sweep", {"model.name": ["persistence", "lstm"]})
    combinations = iter(itertools.product(*sweep.values()))
    requested = 1
    for choices in sweep.values():
        requested *= len(choices)
    deadline = time.monotonic() + cfg.get("budget_seconds", 60)
    results, running, free = [], {}, list(devices)
    exhausted = False
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        while running or not exhausted:
            while free and time.monotonic() < deadline and not exhausted:
                try:
                    combination = next(combinations)
                except StopIteration:
                    exhausted = True
                    break
                current = copy.deepcopy(cfg)
                current.pop("sweep", None)
                for key, value in zip(sweep, combination):
                    set_nested(current, key, value)
                device = free.pop(0)
                current["model"]["device"] = device
                remaining = deadline - time.monotonic()
                timeout = min(remaining, settings.get("job_timeout_seconds", remaining))
                running[executor.submit(isolated_run, current, timeout, resume)] = device
            if time.monotonic() >= deadline:
                exhausted = True
            if not running:
                break
            done, _ = wait(running, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                free.append(running.pop(future))
                result = future.result()
                results.append(result)
                print(f"{result['run_id']}: {result['status']}", flush=True)
    summary = {
        "requested": requested,
        "processed": len(results),
        "completed": sum(r["status"] == "completed" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "interrupted": sum(r["status"] == "interrupted" for r in results),
        "unlaunched": requested - len(results),
        "budget_seconds": cfg.get("budget_seconds", 60),
        "devices": devices,
        "budget_policy": "isolated hard wall-clock jobs; up to 3 seconds graceful termination",
        "run_ids": [r["run_id"] for r in results],
    }
    Path(cfg["output"]).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(cfg["output"]) / "benchmark.json", summary)
    return summary
