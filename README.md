# Intelligent Traffic Management System

Multi-module system that detects and counts vehicles with **YOLOv8 + SORT**, then uses those counts to compute adaptive traffic-light green times. A **Pygame** junction simulation visualizes signal phases and vehicle flow for demos and validation.

## Project overview

Core functionality:

1. **Vehicle detection** — YOLOv8 inference on junction video/GIF with optional ROI mask  
2. **Multi-object tracking** — SORT assigns stable IDs across frames  
3. **Line-crossing counts** — per-lane tallies for cars, trucks, buses, and motorbikes  
4. **Adaptive timing** — green-time allocation weighted by lane density and vehicle class  
5. **Simulation** — four-way intersection with queue-aware signal switching  

## Repository structure

```text
SIH_Hackathon/
├── assets/                      # Centralized media, maps, signals, weights
│   ├── images/                  # Intersection map + vehicle/signal sprites
│   ├── media/                   # Test videos / demo GIF / masks
│   ├── models/                  # YOLO weights (yolov8n.pt)
│   └── output/                  # Sample screenshots
├── config/
│   ├── detection/               # Confidence, count lines, adaptive timing
│   └── simulation/              # Spawn rates, signal timings, window settings
├── src/traffic_management/
│   ├── detection/               # YOLOv8 counter + SORT tracker
│   ├── logic/                   # Adaptive green-time algorithms
│   ├── simulation/              # Pygame sim (+ legacy reference)
│   └── utils/                   # Path / config helpers
├── tests/
│   ├── unit/                    # Logic and path unit tests
│   ├── integration/             # Import / wiring smoke tests
│   └── validation/              # End-to-end detection & simulation reports
├── docs/                        # Project site + validation JSON reports
├── run_detection.py             # CLI entry for detection
├── run_simulation.py            # CLI entry for simulation
└── requirements.txt
```

## Local setup

### Prerequisites

- Python 3.10+ (3.11–3.13 recommended)
- Git
- Optional: CUDA-capable GPU for faster YOLO inference

### Install

```bash
git clone https://github.com/jambhaleAnuj/Traffic_signal_counter_using_car_count_python.git
cd SIH_Hackathon   # or your local clone folder name

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Key libraries:

| Package | Role |
| --- | --- |
| `ultralytics` / `torch` | YOLOv8 detection |
| `opencv-python` | Video I/O and overlays |
| `filterpy` / `lap` / `scipy` | SORT tracking |
| `pygame` | Intersection simulation |
| `PyYAML` | Config loading |

Default weights ship at `assets/models/yolov8n.pt`. On first Ultralytics use, additional caches may download automatically.

### Optional test videos

Place clips under `assets/media/`:

- `sample_video.mp4` — default highway clip
- `sample_video_2.mp4` — second urban junction clip
- `cars4.mp4` — optional extra clip for the legacy config

## Usage

Run all commands from the repository root so `src.*` imports resolve.

### 1. YOLOv8 vehicle detection

Headless validation-style run (prints a JSON report):

```bash
python run_detection.py --max-frames 60
```

Interactive annotated window:

```bash
python run_detection.py --display
python run_detection.py --display --config config/detection/sample_video_2.yaml
```

The preview window is capped at **960×540** so 720p/1080p clips do not fill the screen. Override with `--display-max-width` / `--display-max-height`.

Useful flags:

```bash
python run_detection.py --config config/detection/default.yaml
python run_detection.py --video assets/media/sample_video.mp4 --conf 0.3
python run_detection.py --weights assets/models/yolov8n.pt --report docs/validation/manual_detection.json
```

Second sample video:

```bash
python run_detection.py --display --config config/detection/sample_video_2.yaml
```

### 2. Traffic simulation

Windowed demo (stops after N frames; default 900):

```bash
python run_simulation.py
```

Headless / CI-friendly run:

```bash
python run_simulation.py --headless --max-frames 900 --report docs/validation/manual_simulation.json
```

### 3. Adjust system parameters

Edit YAML rather than hard-coded values.

**Detection** — `config/detection/default.yaml`

- `confidence_threshold` — minimum YOLO confidence  
- `count_line` / lane `count_line` — crossing geometry `[x1, y1, x2, y2]`  
- `adaptive_timing.*` — base / min / max green times and lane weights  
- `signal.*` — simple threshold-based timer overrides  

**Simulation** — `config/simulation/default.yaml`

- `spawn.interval_frames` / `max_active_vehicles`  
- `signal.base_green_time`, `yellow_time`, min/max green  
- `adaptive_timing.enabled` and per-lane weights  

## Traffic management logic

Adaptive timing lives in `src/traffic_management/logic/adaptive_timing.py`.

1. Each counted vehicle contributes a class weight:

   | Class | Weight |
   | --- | --- |
   | bike / motorbike | 0.5 |
   | car | 1.0 |
   | truck | 2.0 |
   | bus | 3.0 |

2. Lane density = Σ(class weight × count) × lane weight.  
3. Each lane receives a share of green time proportional to density.  
4. Lanes with >40% of total density get an extra `traffic_weight_multiplier` boost (capped).  
5. Final green time is clamped to `[min_green_time, max_green_time]`.  

Detection path: YOLOv8 → SORT IDs → line crossing → per-lane sets → `calculate_adaptive_green_times()`.

Simulation path: synthetic spawns build queues; when a phase ends, queue composition drives the next green duration for the active approach.

## Simulation assessment & limitations

Validated findings for the Pygame environment:

| Issue | Status |
| --- | --- |
| Infinite main loop with no test mode | Fixed via `--max-frames` / `--headless` |
| Hard-coded asset paths after reorg | Fixed via `utils.paths` + `assets/` |
| Detector counts not wired into live sim timings | Partially addressed: sim uses **queue-based** adaptive timing, not a live YOLO feed |
| No collision / lane-change physics | Still limited — vehicles move independently |
| Legacy dual implementations | Kept as `legacy_pygame_simulation.py` for reference |
| Headless CI validation | Supported (`SDL_VIDEODRIVER=dummy`) |

### Planned improvements

- Stream detector lane counts into the simulation over a shared queue or socket  
- Add gap-based car-following and stop-line queue spacing  
- Multi-junction scenarios and configurable maps  
- Deterministic seeded runs for reproducible demos  

## Tests & validation

Unit / integration:

```bash
python -m pytest tests/unit tests/integration -q
```

End-to-end validation (writes JSON under `docs/validation/`):

```bash
python tests/validation/validate_detection.py
python tests/validation/validate_simulation.py
```

Reports:

- `docs/validation/detection_report.json`  
- `docs/validation/simulation_report.json`  

## Contribution guidelines

1. Fork the repo and create a feature branch (`feature/…` or `fix/…`).  
2. Keep modules under the existing `src/traffic_management/{detection,logic,simulation,utils}` layout.  
3. Put new tunables in `config/` YAML — avoid scattering magic numbers.  
4. Add or update tests under `tests/unit` or `tests/validation`.  
5. Run unit tests and at least one validation script before opening a PR.  
6. Prefer small, focused PRs with a short summary of behavior change and how you tested it.  
7. Do not commit large raw videos or alternate weight dumps; document them under `assets/*/README.md` instead.  

Bug reports and feature ideas: use the GitHub issue templates under `.github/ISSUE_TEMPLATE/`.

## Version history

| Version | Date | Notes |
| --- | --- | --- |
| 0.1.0 | 2025-08-19 | Initial ATSM release (YOLOv8 + SORT counting, Zenodo DOI) |
| 0.2.0 | 2026-08-21 | Repository restructure (`src` / `assets` / `config` / `tests`), validation CLIs, adaptive timing module, README refresh |

## License

This project is licensed under the **GNU General Public License v3.0 only** (`GPL-3.0-only`). See [`LICENSE`](LICENSE).

If you use this software in research, please cite the repository (see [`CITATION.cff`](CITATION.cff)):

> Adaptive Traffic Signal Management (ATSM): Vehicle Counting with YOLOv8 + SORT  
> DOI: https://doi.org/10.5281/zenodo.16903140

## Quick reference

| Goal | Command |
| --- | --- |
| Detect vehicles | `python run_detection.py --display` |
| Headless detection sample | `python run_detection.py --max-frames 60` |
| Launch simulation | `python run_simulation.py` |
| Headless simulation | `python run_simulation.py --headless --max-frames 900` |
| Unit tests | `python -m pytest tests/unit tests/integration -q` |
| Validate detection | `python tests/validation/validate_detection.py` |
| Validate simulation | `python tests/validation/validate_simulation.py` |

Detailed findings from the verification pass are in [`docs/validation/FINDINGS.md`](docs/validation/FINDINGS.md).
