import sys
from pathlib import Path

# Ensure the repo root is importable so tests can reach both `src` (installed)
# and top-level `scripts` (not packaged).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
