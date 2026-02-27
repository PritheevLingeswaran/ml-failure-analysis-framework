from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from evaluation.evaluate import run_evaluate

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    run_evaluate(cfg)

if __name__ == "__main__":
    main()
