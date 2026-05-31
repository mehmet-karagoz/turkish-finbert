from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"


def ensure_project_dirs() -> None:
    for path in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "labels", REPORT_DIR, FIGURE_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)

