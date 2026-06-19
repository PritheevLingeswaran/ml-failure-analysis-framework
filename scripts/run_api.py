from __future__ import annotations
import argparse
import uvicorn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.api.app import create_app

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg["api"]["host"],
        port=int(cfg["api"]["port"]),
        reload=bool(cfg["api"].get("reload", False)),
    )

if __name__ == "__main__":
    main()
