"""
config.py — Single source of truth for all paths and settings.
Place in: Scripts/config.py
All other scripts import from here — no hardcoded paths anywhere else.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────
# Walks up from Scripts/ to find Project 1/.env
_here = Path(__file__).resolve().parent        # Scripts/
_root = _here.parent                           # Project 1/
load_dotenv(_root / ".env")

# ── Project structure ─────────────────────────────────────
PROJECT_ROOT = _root
SCRIPTS_DIR  = _here
SAR_DATA     = PROJECT_ROOT / "sar_data"

# ── Sub-paths ─────────────────────────────────────────────
MODELS_DIR    = SAR_DATA / "models"
DATASET_DIR   = SAR_DATA / "dataset"
REAL_DIR      = SAR_DATA / "dataset_real"
THERMAL_DIR   = SAR_DATA / "dataset_thermal"
LOGS_DIR      = SAR_DATA / "logs"
RAW_DIR       = SAR_DATA / "raw"
ANNOTS_DIR    = SAR_DATA / "annotations" / "all"
SYNTHETIC_DIR = SAR_DATA / "synthetic"
FUSED_DIR     = SAR_DATA / "fused"

# ── Raw source folders ────────────────────────────────────
RAW_RGB_REAL      = RAW_DIR / "rgb" / "real"
RAW_RGB_ONLINE    = RAW_DIR / "rgb" / "online"
RAW_RGB_CUSTOM    = RAW_DIR / "rgb" / "custom"
RAW_THERMAL_REAL  = RAW_DIR / "thermal" / "real"

# ── Task script folders (for sys.path) ───────────────────
TASK_FOLDERS = [
    str(SCRIPTS_DIR / f"task {i}") for i in range(1, 6)
] + [str(SCRIPTS_DIR)]


def setup_path():
    """Call this at top of any script that imports across tasks."""
    for folder in TASK_FOLDERS:
        if folder not in sys.path:
            sys.path.insert(0, folder)


# ── Model path finder ─────────────────────────────────────
def find_best_model(subdir):
    """Auto-finds best.pt in a model subdirectory."""
    d = MODELS_DIR / subdir
    if not d.exists():
        return None
    matches = sorted(d.rglob("best.pt"))
    return matches[-1] if matches else None   # most recent run


# ── Kaggle credentials (from .env, never hardcoded) ───────
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY", "")

# ── Project metadata ──────────────────────────────────────
PROJECT_AUTHOR      = os.getenv("PROJECT_AUTHOR", "Unknown")
PROJECT_INSTITUTION = os.getenv("PROJECT_INSTITUTION", "IISc")
PROJECT_YEAR        = os.getenv("PROJECT_YEAR", "2026")


if __name__ == "__main__":
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"sar_data          : {SAR_DATA}")
    print(f"RGB model         : {find_best_model('rgb_detector')}")
    print(f"Real human model  : {find_best_model('human_detector_real')}")
    print(f"Kaggle user       : {KAGGLE_USERNAME or 'NOT SET'}")
    print(f"Author            : {PROJECT_AUTHOR}")