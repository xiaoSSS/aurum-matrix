from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep tests isolated from developer-local aurum.config.json secrets and cache.
os.environ.setdefault("AURUM_CONFIG_FILE", str(ROOT / "tests" / "_no_config_.json"))
