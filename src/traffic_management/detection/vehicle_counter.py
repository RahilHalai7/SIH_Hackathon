from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.traffic_management.detection.sort_tracker import Sort
from src.traffic_management.logic.adaptive_timing import (
    calculate_adaptive_green_times,
    check_line_crossing,
)
from src.traffic_management.utils.paths import detection_default_config, resolve_repo_path


DEFAULT_CONFIG = {
    "video_path": "assets/media/sample_video.mp4",
    "mask_path": "",
    "weights_path": "assets/models/yolov8n.pt",
    "classes_to_count": ["car", "truck", "motorbike", "bus"],
    "confidence_threshold": 0.3,
    "count_line": [90, 510, 1210, 510],
    "count_line_tolerance": 28,
    "display": {"max_width": 960, "max_height": 540},
    "tracker": {"max_age": 20, "min_hits": 3, "iou_threshold": 0.3},
    "signal": {
        "car_count_threshold": 10,
        "normal_red_timer": 60,
        "reduced_red_timer": 30,
        "cooldown_duration": 10,
    },
    "adaptive_timing": {
        "enabled": True,
        "lanes": [
            {"name": "Left", "count_line": [90, 510, 590, 510], "weight": 1.0},
            {"name": "Right", "count_line": [650, 510, 1210, 510], "weight": 1.0},
        ],
        "base_green_time": 30,
        "max_green_time": 60,
        "min_green_time": 15,
        "traffic_weight_multiplier": 2.0,
        "update_interval": 5,
    },
}

COCO_NAME_ALIASES = {"motorcycle": "motorbike"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 vehicle detection and counting.")
    parser.add_argument("--config", default=str(detection_default_config()), help="Path to YAML config file")
    parser.add_argument("--video", dest="video_path", help="Override the input video path")
    parser.add_argument("--mask", dest="mask_path", help="Override the mask image path")
    parser.add_argument("--weights", dest="weights_path", help="Override the YOLO weight path")
    parser.add_argument("--conf", dest="confidence_threshold", type=float, help="Override the confidence threshold")
    parser.add_argument("--display", action="store_true", help="Show the annotated video window")
    parser.add_argument("--display-max-width", type=int, help="Max preview window width in pixels")
    parser.add_argument("--display-max-height", type=int, help="Max preview window height in pixels")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means run full input")
    parser.add_argument("--report", type=str, help="Write a JSON validation report to this path")
    return parser.parse_args()


def load_config(path: str | Path | None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if not path:
        return config

    config_path = resolve_repo_path(path)
    if config_path is None or not config_path.exists():
        return config

    with config_path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}

    config |= user_config
    for section in ("tracker", "signal", "adaptive_timing", "display"):
        if section in user_config:
            config[section] = DEFAULT_CONFIG.get(section, {}) | user_config[section]
    return config


def scale_display_frame(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def normalise_class_name(name: str) -> str:
    return COCO_NAME_ALIASES.get(name, name)


def load_model(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    return YOLO(str(weights_path))


def load_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Video source could not be opened: {video_path}")
    return capture


def match_track_categories(tracked_objects: np.ndarray, detections: list[dict]) -> dict[int, str]:
    matched: dict[int, str] = {}
    if len(tracked_objects) == 0 or not detections:
        return matched

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)
        best_iou = 0.0
        best_class = "car"
        track_box = np.array([x1, y1, x2, y2], dtype=float)
        for detection in detections:
            det_box = detection["bbox"][:4]
            xx1 = max(track_box[0], det_box[0])
            yy1 = max(track_box[1], det_box[1])
            xx2 = min(track_box[2], det_box[2])
            yy2 = min(track_box[3], det_box[3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            track_area = max(1.0, (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]))
            det_area = max(1.0, (det_box[2] - det_box[0]) * (det_box[3] - det_box[1]))
            union = track_area + det_area - inter
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_class = detection["class_name"]
        matched[int(obj_id)] = best_class
    return matched


def draw_overlay(frame: np.ndarray, report: dict, current_lane: str, remaining_time: int) -> None:
    y = 30
    cv2.putText(frame, f"Active lane: {current_lane}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30
    cv2.putText(frame, f"Timer: {remaining_time}s", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y += 30
    cv2.putText(
        frame,
        f"Frames: {report['frames_processed']} Detections: {report['total_detections']}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    for category, count in report["vehicle_counts"].items():
        y += 28
        cv2.putText(frame, f"{category}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


def build_report() -> dict:
    return {
        "frames_processed": 0,
        "total_detections": 0,
        "tracked_objects_peak": 0,
        "vehicle_counts": {"car": 0, "truck": 0, "motorbike": 0, "bus": 0},
        "lane_counts": {},
        "adaptive_green_times": {},
        "errors": [],
        "warnings": [],
        "runtime_seconds": 0.0,
        "fps": 0.0,
    }


def run_detection(config: dict, display: bool = False, max_frames: int = 0) -> dict:
    report = build_report()
    start_time = time.time()

    video_path = resolve_repo_path(config["video_path"])
    mask_path = resolve_repo_path(config.get("mask_path"))
    weights_path = resolve_repo_path(config["weights_path"])

    try:
        capture = load_capture(video_path)
        model = load_model(weights_path)
    except Exception as exc:
        report["errors"].append(str(exc))
        return report

    region_mask = None
    if mask_path and mask_path.exists():
        region_mask = cv2.imread(str(mask_path))
    elif mask_path:
        report["warnings"].append(f"Mask not found: {mask_path}")

    tracker_cfg = config.get("tracker", {})
    tracker = Sort(
        max_age=int(tracker_cfg.get("max_age", 20)),
        min_hits=int(tracker_cfg.get("min_hits", 3)),
        iou_threshold=float(tracker_cfg.get("iou_threshold", 0.3)),
    )

    lanes = config.get("adaptive_timing", {}).get("lanes", [])
    if not lanes:
        lanes = [{"name": "Main", "count_line": config["count_line"], "weight": 1.0}]

    lane_vehicle_counts = {
        lane["name"]: {"car": set(), "truck": set(), "motorbike": set(), "bus": set()} for lane in lanes
    }
    report["lane_counts"] = {lane["name"]: 0 for lane in lanes}
    classes_to_count = {normalise_class_name(name) for name in config.get("classes_to_count", [])}
    conf_threshold = float(config.get("confidence_threshold", 0.3))
    display_cfg = config.get("display", {})
    max_display_width = int(display_cfg.get("max_width", 960))
    max_display_height = int(display_cfg.get("max_height", 540))
    crossing_tolerance = int(config.get("count_line_tolerance", 28))
    lane_index = 0
    last_lane_switch = time.time()
    display_window_ready = False

    frame_number = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break

        frame_number += 1
        if max_frames and frame_number > max_frames:
            break

        report["frames_processed"] += 1
        masked_frame = frame
        if region_mask is not None:
            if frame.shape[:2] == region_mask.shape[:2]:
                masked_frame = cv2.bitwise_and(frame, region_mask)
            elif "Mask/frame dimension mismatch." not in report["warnings"]:
                report["warnings"].append("Mask/frame dimension mismatch.")

        raw_results = model.predict(masked_frame, verbose=False)
        detections_array = np.empty((0, 5))
        detections: list[dict] = []
        model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

        for result in raw_results:
            for box in result.boxes:
                class_name = normalise_class_name(model_names[int(box.cls[0])])
                confidence = float(box.conf[0])
                if class_name not in classes_to_count or confidence < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections_array = np.vstack((detections_array, np.array([x1, y1, x2, y2, confidence])))
                detections.append({"bbox": np.array([x1, y1, x2, y2, confidence]), "class_name": class_name})

        tracked_objects = tracker.update(detections_array)
        report["total_detections"] += len(detections)
        report["tracked_objects_peak"] = max(report["tracked_objects_peak"], len(tracked_objects))
        track_categories = match_track_categories(tracked_objects, detections)

        for lane in lanes:
            x1, y1, x2, y2 = lane["count_line"]
            color = (0, 255, 0) if lane["name"] == lanes[lane_index]["name"] else (255, 0, 0)
            if display:
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, lane["name"], (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_tolerance = max(crossing_tolerance, frame.shape[0] // 30)
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj.astype(int)
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            category = track_categories.get(int(obj_id), "car")

            if display:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{obj_id}:{category}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            for lane in lanes:
                if not check_line_crossing(center_x, center_y, lane["count_line"], tolerance=frame_tolerance):
                    continue
                counted_set = lane_vehicle_counts[lane["name"]][category]
                if obj_id in counted_set:
                    continue
                counted_set.add(obj_id)
                report["vehicle_counts"][category] += 1
                report["lane_counts"][lane["name"]] += 1

        adaptive_cfg = config.get("adaptive_timing", {})
        green_times = calculate_adaptive_green_times(lane_vehicle_counts, adaptive_cfg) or {}
        report["adaptive_green_times"] = green_times

        current_lane = lanes[lane_index]["name"]
        current_green_time = green_times.get(current_lane, adaptive_cfg.get("base_green_time", 30))
        elapsed = int(time.time() - last_lane_switch)
        remaining_time = max(0, current_green_time - elapsed)
        if remaining_time == 0:
            lane_index = (lane_index + 1) % len(lanes)
            last_lane_switch = time.time()

        if display:
            draw_overlay(frame, report, current_lane, remaining_time)
            preview = scale_display_frame(frame, max_display_width, max_display_height)
            if not display_window_ready:
                cv2.namedWindow("Vehicle Counter", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Vehicle Counter", preview.shape[1], preview.shape[0])
                cv2.moveWindow("Vehicle Counter", 40, 40)
                display_window_ready = True
            cv2.imshow("Vehicle Counter", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if display:
        cv2.destroyAllWindows()

    report["runtime_seconds"] = round(time.time() - start_time, 3)
    if report["runtime_seconds"] > 0:
        report["fps"] = round(report["frames_processed"] / report["runtime_seconds"], 2)
    if report["total_detections"] == 0:
        report["warnings"].append("No qualifying vehicles were detected in the processed frames.")
    return report


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.video_path:
        config["video_path"] = args.video_path
    if args.mask_path:
        config["mask_path"] = args.mask_path
    if args.weights_path:
        config["weights_path"] = args.weights_path
    if args.confidence_threshold is not None:
        config["confidence_threshold"] = args.confidence_threshold
    if args.display_max_width:
        config.setdefault("display", {})["max_width"] = args.display_max_width
    if args.display_max_height:
        config.setdefault("display", {})["max_height"] = args.display_max_height

    report = run_detection(config, display=args.display, max_frames=args.max_frames)
    if args.report:
        report_path = resolve_repo_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
