from src.traffic_management.utils.paths import (
    REPO_ROOT,
    detection_default_config,
    resolve_repo_path,
    simulation_default_config,
)


def test_repo_root_exists():
    assert REPO_ROOT.exists()
    assert (REPO_ROOT / "src").exists()
    assert (REPO_ROOT / "assets").exists()
    assert (REPO_ROOT / "config").exists()


def test_resolve_repo_path_handles_empty_and_relative():
    assert resolve_repo_path(None) is None
    assert resolve_repo_path("") is None
    assert resolve_repo_path("   ") is None
    resolved = resolve_repo_path("assets/models/yolov8n.pt")
    assert resolved == REPO_ROOT / "assets" / "models" / "yolov8n.pt"
    assert resolved.exists()


def test_default_configs_exist():
    assert detection_default_config().exists()
    assert simulation_default_config().exists()
