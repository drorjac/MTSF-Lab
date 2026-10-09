import sys
from dynforecast.cli import main

raise SystemExit(main(["run", *sys.argv[1:]]))
