"""Entry point for: python -m mps_sdpa.cli"""
import sys

from . import cli

if __name__ == "__main__":
    sys.exit(cli.main())
