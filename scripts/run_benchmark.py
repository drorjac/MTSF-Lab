import sys
from dynforecast.cli import main

args = sys.argv[1:]
experiments = [arg for arg in args if arg.startswith("experiment=")]
if experiments:
    args = [arg for arg in args if not arg.startswith("experiment=")]
    args = ["--config", f"configs/experiments/{experiments[0].split('=', 1)[1]}.yaml", *args]
raise SystemExit(main(["benchmark", *args]))
