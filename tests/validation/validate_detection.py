from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.traffic_management.detection.vehicle_counter import load_config, run_detection
from src.traffic_management.utils.paths import resolve_repo_path


def main() -> int:
    configs = {
        "default": REPO_ROOT / "config" / "detection" / "default.yaml",
        "sample_video_2": REPO_ROOT / "config" / "detection" / "sample_video_2.yaml",
        "cars4": REPO_ROOT / "config" / "detection" / "cars4.yaml",
    }
    frame_limits = {"default": 45, "sample_video_2": 20, "cars4": 10}
    results: dict[str, dict] = {}
    exit_code = 0

    for name, config_path in configs.items():
        config = load_config(config_path)
        video_path = resolve_repo_path(config.get("video_path"))
        if video_path is None or not video_path.exists():
            results[name] = {
                "frames_processed": 0,
                "total_detections": 0,
                "errors": [f"Test video missing: {config.get('video_path')}"],
                "warnings": [
                    "Optional clip is not present; skip this validation path."
                    if name != "default"
                    else "Required sample video is missing."
                ],
                "skipped": True,
            }
            if name == "default":
                exit_code = 1
            continue

        report = run_detection(config, display=False, max_frames=frame_limits[name])
        results[name] = report
        if report.get("errors"):
            exit_code = 1

    output_dir = REPO_ROOT / "docs" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detection_report.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
