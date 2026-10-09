import sys
from dynforecast.cli import main

raise SystemExit(main(["download", *sys.argv[1:]]))
