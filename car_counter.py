import argparse
import time
import cv2
import math
import cvzone
import numpy as np
import yaml
from sort import Sort
from ultralytics import YOLO


def load_config(path: str | None) -> dict:
    default_cfg = {
        "video_path": "Media/cars2.mp4",
        "mask_path": "Media/mask.png",
        "weights_path": "Weights/yolov8n.pt",
        "classes_to_count": ["car", "truck", "motorbike", "bus"],
        "confidence_threshold": 0.3,
        "count_line": [199, 363, 1208, 377],
        "tracker": {"max_age": 20, "min_hits": 3, "iou_threshold": 0.3},
        "signal": {
            "car_count_threshold": 10,
            "normal_red_timer": 60,
            "reduced_red_timer": 30,
            "cooldown_duration": 10,
        },
    }
    if not path:
        return default_cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # merge shallow
        cfg = default_cfg | user_cfg
        # nested merges
        if "tracker" in user_cfg:
            cfg["tracker"] = default_cfg["tracker"] | user_cfg["tracker"]
        if "signal" in user_cfg:
            cfg["signal"] = default_cfg["signal"] | user_cfg["signal"]
        return cfg
    except FileNotFoundError:
        return default_cfg


def correct_classification_by_size(class_name, bbox, size_thresholds):
    """
    Correct vehicle classification based on bounding box size.
    Helps distinguish between cars and buses when YOLO is uncertain.
    """
    x1, y1, x2, y2 = bbox[:4]
    area = (x2 - x1) * (y2 - y1)
    
    # Check if the detected class matches its expected size range
    if class_name in size_thresholds:
        min_area = size_thresholds[class_name]['min_area']
        max_area = size_thresholds[class_name]['max_area']
        
        if area < min_area or area > max_area:
            # Find the most likely class based on size
            for vehicle_type, thresholds in size_thresholds.items():
                if thresholds['min_area'] <= area <= thresholds['max_area']:
                    return vehicle_type
    
    return class_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ATSM Vehicle Counter (YOLOv8 + SORT)")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    p.add_argument("--video", dest="video_path", type=str, help="Override: video path")
    p.add_argument("--mask", dest="mask_path", type=str, help="Override: mask path")
    p.add_argument("--weights", dest="weights_path", type=str, help="Override: YOLO weights path")
    p.add_argument("--line", nargs=4, type=int, metavar=("x1", "y1", "x2", "y2"), help="Override: count line coords")
    p.add_argument("--threshold", type=int, dest="car_count_threshold", help="Override: congestion threshold")
    p.add_argument("--normal", type=int, dest="normal_red_timer", help="Override: normal red timer (s)")
    p.add_argument("--reduced", type=int, dest="reduced_red_timer", help="Override: reduced red timer (s)")
    p.add_argument("--cooldown", type=int, dest="cooldown_duration", help="Override: cooldown duration (s)")
    p.add_argument("--conf", type=float, dest="confidence_threshold", help="Override: detection confidence threshold")
    p.add_argument("--car-threshold", type=int, help="Override: car count threshold")
    p.add_argument("--truck-threshold", type=int, help="Override: truck count threshold")
    p.add_argument("--bus-threshold", type=int, help="Override: bus count threshold")
    p.add_argument("--motorbike-threshold", type=int, help="Override: motorbike count threshold")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides if provided
    if args.video_path: cfg["video_path"] = args.video_path
    if args.mask_path: cfg["mask_path"] = args.mask_path
    if args.weights_path: cfg["weights_path"] = args.weights_path
    if args.line: cfg["count_line"] = list(args.line)
    if args.car_count_threshold is not None: cfg["signal"]["car_count_threshold"] = args.car_count_threshold
    if args.normal_red_timer is not None: cfg["signal"]["normal_red_timer"] = args.normal_red_timer
    if args.reduced_red_timer is not None: cfg["signal"]["reduced_red_timer"] = args.reduced_red_timer
    if args.cooldown_duration is not None: cfg["signal"]["cooldown_duration"] = args.cooldown_duration
    if args.confidence_threshold is not None: cfg["confidence_threshold"] = args.confidence_threshold
    
    # Handle category-specific threshold overrides
    if not cfg.get("category_thresholds"):
        cfg["category_thresholds"] = {}
    if args.car_threshold is not None: cfg["category_thresholds"]["car"] = args.car_threshold
    if args.truck_threshold is not None: cfg["category_thresholds"]["truck"] = args.truck_threshold
    if args.bus_threshold is not None: cfg["category_thresholds"]["bus"] = args.bus_threshold
    if args.motorbike_threshold is not None: cfg["category_thresholds"]["motorbike"] = args.motorbike_threshold

    # Initialize video capture
    video_path = cfg["video_path"]
    cap = cv2.VideoCapture(video_path)

    # Load YOLO model with custom weights
    yolo_model = YOLO(cfg["weights_path"])

    # Define class names
    class_labels = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # Load region mask
    region_mask = cv2.imread(cfg["mask_path"]) if cfg.get("mask_path") else None

    # Initialize tracker
    tracker_params = cfg.get("tracker", {})
    tracker = Sort(
        max_age=int(tracker_params.get("max_age", 20)),
        min_hits=int(tracker_params.get("min_hits", 3)),
        iou_threshold=float(tracker_params.get("iou_threshold", 0.3)),
    )

    # Define line limits for counting
    count_line = cfg["count_line"]

    # Dictionary to track counted vehicles by category
    counted_vehicles = {
        'car': set(),
        'truck': set(), 
        'motorbike': set(),
        'bus': set()
    }

    # Detection filter
    classes_to_count = set(cfg.get("classes_to_count", ["car", "truck", "motorbike", "bus"]))
    conf_thresh = float(cfg.get("confidence_threshold", 0.5))
    
    # Size-based classification thresholds (in pixels)
    size_thresholds = cfg.get("size_thresholds", {
        'car': {'min_area': 1000, 'max_area': 50000},      # Small to medium vehicles
        'bus': {'min_area': 15000, 'max_area': 100000},    # Large vehicles
        'truck': {'min_area': 8000, 'max_area': 80000},    # Medium to large vehicles
        'motorbike': {'min_area': 500, 'max_area': 15000}  # Small vehicles
    })

    # Signal thresholds
    sig = cfg.get("signal", {})
    car_count_threshold = int(sig.get("car_count_threshold", 10))
    normal_red_timer = int(sig.get("normal_red_timer", 60))
    reduced_red_timer = int(sig.get("reduced_red_timer", 30))
    cooldown_duration = int(sig.get("cooldown_duration", 10))
    
    # Category-specific thresholds (optional)
    category_thresholds = cfg.get("category_thresholds", {})

    # Initialize timers
    red_light_timer = normal_red_timer
    last_timer_update_time = time.time()

    # Variables to track reductions and cooldowns
    timer_reduced_message = ""
    reduction_count = 0
    cooldown_active = False
    cooldown_timer_start = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if region_mask is not None:
            masked_frame = cv2.bitwise_and(frame, region_mask)
        else:
            masked_frame = frame

        # Perform object detection
        detection_results = yolo_model(masked_frame, stream=True)

        # Collect detections
        detection_array = np.empty((0, 5))

        # Store detection info for categorization
        detection_info = []
        
        for result in detection_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                confidence = math.ceil((box.conf[0] * 100)) / 100
                class_id = int(box.cls[0])
                class_name = class_labels[class_id]

                if class_name in classes_to_count and confidence > conf_thresh:
                    # Apply size-based classification correction
                    corrected_class = correct_classification_by_size(class_name, [x1, y1, x2, y2], size_thresholds)
                    
                    detection_entry = np.array([x1, y1, x2, y2, confidence])
                    detection_array = np.vstack((detection_array, detection_entry))
                    detection_info.append({
                        'bbox': detection_entry,
                        'class_name': corrected_class,
                        'confidence': confidence,
                        'original_class': class_name
                    })
        tracked_objects = tracker.update(detection_array)

        # Draw count line
        cv2.line(frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (0, 255, 0), 2)

        # Create mapping between tracked objects and their categories
        tracked_categories = {}
        
        # Match tracked objects with detection info to get categories
        for i, obj in enumerate(tracked_objects):
            if i < len(detection_info):
                tracked_categories[int(obj[4])] = detection_info[i]['class_name']

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            width, height = x2 - x1, y2 - y1
            obj_id = int(obj_id)
            
            # Get vehicle category for this tracked object
            vehicle_category = tracked_categories.get(obj_id, 'car')  # Default to car if not found

            # Draw bounding boxes and labels with category info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Show original vs corrected classification if different
            if i < len(detection_info) and detection_info[i].get('original_class') != vehicle_category:
                cv2.putText(frame, f'ID: {obj_id} ({detection_info[i]["original_class"]}->{vehicle_category})', 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f'ID: {obj_id} ({vehicle_category})', 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Calculate center of the box
            center_x, center_y = x1 + width // 2, y1 + height // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Check if vehicle crosses the counting line
            if count_line[0] < center_x < count_line[2] and count_line[1] - 20 < center_y < count_line[1] + 20:
                if obj_id not in counted_vehicles[vehicle_category]:
                    counted_vehicles[vehicle_category].add(obj_id)
                    cv2.line(frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (255, 0, 0), 2)

        # Calculate total vehicle count for threshold checking
        total_vehicle_count = sum(len(vehicles) for vehicles in counted_vehicles.values())
        
        # Adjust red light timer if total vehicle count exceeds threshold
        current_time = time.time()
        if total_vehicle_count > car_count_threshold:
            if not cooldown_active:
                if reduction_count < 2 and red_light_timer != reduced_red_timer:
                    timer_reduced_message = f"Timer reduced by {normal_red_timer - reduced_red_timer} seconds"
                    red_light_timer = reduced_red_timer
                    reduction_count += 1
        else:
            timer_reduced_message = ""
            red_light_timer = normal_red_timer
            reduction_count = 0  # Reset reduction count when below threshold

        # Cooldown logic
        if cooldown_active:
            elapsed_cooldown_time = current_time - (cooldown_timer_start or current_time)
            if elapsed_cooldown_time >= cooldown_duration:
                cooldown_active = False

        if reduction_count >= 2 and not cooldown_active:
            cooldown_active = True
            cooldown_timer_start = current_time

        # Calculate remaining timer
        elapsed_time = int(current_time - last_timer_update_time)
        remaining_time = max(0, red_light_timer - elapsed_time)

        if remaining_time == 0:
            last_timer_update_time = current_time
            red_light_timer = normal_red_timer
            timer_reduced_message = ""  # Clear message on reset
            # Reset all vehicle counts per cycle
            for category in counted_vehicles:
                counted_vehicles[category].clear()

        # Display categorized vehicle counts, timer, and reduction message
        y_offset = 50
        cvzone.putTextRect(frame, f'TOTAL: {total_vehicle_count}', (20, y_offset), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(255, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX)
        
        # Display individual category counts
        for category, vehicles in counted_vehicles.items():
            y_offset += 40
            cvzone.putTextRect(frame, f'{category.upper()}: {len(vehicles)}', (20, y_offset), scale=0.8, thickness=2, colorT=(255, 255, 255), colorR=(0, 100, 200), font=cv2.FONT_HERSHEY_SIMPLEX)
        
        y_offset += 50
        cvzone.putTextRect(frame, f'TIMER: {remaining_time}s', (20, y_offset), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX)

        if timer_reduced_message:
            y_offset += 40
            cvzone.putTextRect(frame, timer_reduced_message, (20, y_offset), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow("Car Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
