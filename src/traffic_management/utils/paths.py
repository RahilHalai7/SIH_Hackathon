from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets"
CONFIG_DIR = REPO_ROOT / "config"
DOCS_DIR = REPO_ROOT / "docs"


def resolve_repo_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    if isinstance(path_like, str) and not path_like.strip():
        return None
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def detection_default_config() -> Path:
    return CONFIG_DIR / "detection" / "default.yaml"


def simulation_default_config() -> Path:
    return CONFIG_DIR / "simulation" / "default.yaml"


def detection_validation_report() -> Path:
    return DOCS_DIR / "validation" / "detection_report.json"


def simulation_validation_report() -> Path:
    return DOCS_DIR / "validation" / "simulation_report.json"
