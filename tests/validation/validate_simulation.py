from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.traffic_management.simulation.pygame_simulation import TrafficSimulation, load_config


def main() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    config = load_config(str(REPO_ROOT / "config" / "simulation" / "default.yaml"))
    simulation = TrafficSimulation(config, headless=True)
    try:
        report = simulation.run(max_frames=900)
    finally:
        simulation.pygame.quit()

    output_dir = REPO_ROOT / "docs" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulation_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
