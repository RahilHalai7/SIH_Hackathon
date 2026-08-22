from __future__ import annotations

from typing import Iterable


VEHICLE_WEIGHT_MULTIPLIERS = {
    "car": 1.0,
    "motorbike": 0.5,
    "bike": 0.5,
    "truck": 2.0,
    "bus": 3.0,
}


def get_vehicle_weight(vehicle_category: str) -> float:
    return VEHICLE_WEIGHT_MULTIPLIERS.get(vehicle_category, 1.0)


def calculate_vehicle_weight(vehicle_counts: dict[str, Iterable[object]]) -> float:
    total_weighted_count = 0.0
    for category, vehicles in vehicle_counts.items():
        total_weighted_count += len(list(vehicles)) * get_vehicle_weight(category)
    return total_weighted_count


def calculate_adaptive_green_times(
    lane_traffic_data: dict[str, dict[str, Iterable[object]]],
    adaptive_config: dict,
) -> dict[str, int] | None:
    if not adaptive_config.get("enabled", False):
        return None

    lanes = adaptive_config.get("lanes", [])
    base_green_time = adaptive_config.get("base_green_time", 30)
    max_green_time = adaptive_config.get("max_green_time", 60)
    min_green_time = adaptive_config.get("min_green_time", 15)
    traffic_weight_multiplier = adaptive_config.get("traffic_weight_multiplier", 2.0)

    lane_densities: dict[str, float] = {}
    for lane in lanes:
        lane_name = lane["name"]
        counts = lane_traffic_data.get(lane_name, {})
        lane_densities[lane_name] = calculate_vehicle_weight(counts) * lane.get("weight", 1.0)

    total_density = sum(lane_densities.values())
    if total_density == 0:
        return {lane["name"]: base_green_time for lane in lanes}

    adaptive_times: dict[str, int] = {}
    for lane in lanes:
        lane_name = lane["name"]
        lane_density = lane_densities[lane_name]
        if lane_density == 0:
            adaptive_times[lane_name] = min_green_time
            continue

        proportion = lane_density / total_density
        if proportion > 0.4:
            proportion = min(proportion * traffic_weight_multiplier, 1.0)

        green_time = base_green_time + proportion * (max_green_time - base_green_time)
        adaptive_times[lane_name] = int(max(min_green_time, min(max_green_time, green_time)))

    return adaptive_times


def check_line_crossing(center_x: int, center_y: int, count_line: list[int], tolerance: int = 20) -> bool:
    x1, y1, x2, _ = count_line
    return (x1 < center_x < x2) and (y1 - tolerance < center_y < y1 + tolerance)
