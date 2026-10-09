import sys
from dynforecast.cli import main

raise SystemExit(main(["simulate", *sys.argv[1:]]))
