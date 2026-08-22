def test_core_modules_import():
    from src.traffic_management.detection import vehicle_counter
    from src.traffic_management.logic import adaptive_timing
    from src.traffic_management.simulation import pygame_simulation
    from src.traffic_management.utils import paths

    assert callable(vehicle_counter.run_detection)
    assert callable(adaptive_timing.calculate_adaptive_green_times)
    assert callable(pygame_simulation.TrafficSimulation)
    assert paths.REPO_ROOT.exists()
