import sys
from dynforecast.cli import main

raise SystemExit(main(["report", *sys.argv[1:]]))
