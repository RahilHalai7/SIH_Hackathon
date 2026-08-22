from src.traffic_management.logic.adaptive_timing import (
    calculate_adaptive_green_times,
    calculate_vehicle_weight,
    check_line_crossing,
    get_vehicle_weight,
)


def test_vehicle_weights():
    assert get_vehicle_weight("car") == 1.0
    assert get_vehicle_weight("bike") == 0.5
    assert get_vehicle_weight("bus") == 3.0
    assert get_vehicle_weight("unknown") == 1.0


def test_calculate_vehicle_weight():
    counts = {"car": [1, 2], "bus": [1], "bike": [1, 2, 3]}
    assert calculate_vehicle_weight(counts) == 2.0 + 3.0 + 1.5


def test_adaptive_green_times_empty_lanes_use_base():
    adaptive_config = {
        "enabled": True,
        "lanes": [{"name": "North", "weight": 1.0}, {"name": "South", "weight": 1.0}],
        "base_green_time": 30,
        "min_green_time": 15,
        "max_green_time": 60,
        "traffic_weight_multiplier": 2.0,
    }
    times = calculate_adaptive_green_times({"North": {}, "South": {}}, adaptive_config)
    assert times == {"North": 30, "South": 30}


def test_adaptive_green_times_biased_to_heavier_lane():
    adaptive_config = {
        "enabled": True,
        "lanes": [{"name": "North", "weight": 1.0}, {"name": "South", "weight": 1.0}],
        "base_green_time": 30,
        "min_green_time": 15,
        "max_green_time": 60,
        "traffic_weight_multiplier": 2.0,
    }
    lane_traffic = {
        "North": {"bus": list(range(8))},
        "South": {"car": [1]},
    }
    times = calculate_adaptive_green_times(lane_traffic, adaptive_config)
    assert times is not None
    assert times["North"] > times["South"]
    assert 15 <= times["North"] <= 60
    assert 15 <= times["South"] <= 60


def test_adaptive_disabled_returns_none():
    assert calculate_adaptive_green_times({}, {"enabled": False}) is None


def test_line_crossing():
    count_line = [100, 200, 400, 200]
    assert check_line_crossing(250, 200, count_line)
    assert not check_line_crossing(50, 200, count_line)
    assert not check_line_crossing(250, 260, count_line)
