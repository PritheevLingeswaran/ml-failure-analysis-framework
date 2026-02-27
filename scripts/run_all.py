from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cmds = [
        [sys.executable, "scripts/prepare_data.py", "--config", args.config],
        [sys.executable, "scripts/train_models.py", "--config", args.config],
        [sys.executable, "scripts/run_eval.py", "--config", args.config],
    ]
    for c in cmds:
        subprocess.check_call(c)

if __name__ == "__main__":
    main()
