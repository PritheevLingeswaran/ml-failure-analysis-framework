from __future__ import annotations

import argparse
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.api.app import create_app
import uvicorn

def main() -> None:
    parser = argparse.ArgumentParser(prog="ml-failure-analysis-framework")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["api", "eval"], default="eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.mode == "api":
        app = create_app(cfg)
        uvicorn.run(
            app,
            host=cfg["api"]["host"],
            port=int(cfg["api"]["port"]),
            reload=bool(cfg["api"].get("reload", False)),
        )
    else:
        # Delegate to scripts/run_eval.py for richer CLI.
        from evaluation.evaluate import run_evaluate
        run_evaluate(cfg)

if __name__ == "__main__":
    main()
