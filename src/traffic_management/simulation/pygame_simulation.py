from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import yaml

from src.traffic_management.logic.adaptive_timing import calculate_adaptive_green_times
from src.traffic_management.utils.paths import resolve_repo_path, simulation_default_config


VEHICLE_TYPES = ("car", "bus", "truck", "bike")
DIRECTIONS = ("right", "down", "left", "up")
SIGNAL_POSITIONS = {"right": (422, 165), "down": (790, 165), "left": (790, 535), "up": (420, 535)}
SIGNAL_TIMER_POSITIONS = {"right": (422, 145), "down": (790, 145), "left": (790, 515), "up": (420, 515)}
SPAWN_POSITIONS = {
    "right": (-70, 350),
    "down": (640, -70),
    "left": (1070, 430),
    "up": (560, 654),
}
STOP_LINES = {"right": 360, "down": 200, "left": 850, "up": 600}
EXIT_LIMITS = {"right": 1080, "down": 664, "left": -90, "up": -90}
QUEUE_AXES = {"right": "x", "down": "y", "left": "x", "up": "y"}
QUEUE_SIGNS = {"right": 1, "down": 1, "left": -1, "up": -1}
SPEEDS = {"car": 2.2, "bus": 1.7, "truck": 1.6, "bike": 2.6}


@dataclass
class Vehicle:
    vehicle_type: str
    direction: str
    x: float
    y: float
    speed: float
    crossed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pygame traffic simulation.")
    parser.add_argument("--config", default=str(simulation_default_config()), help="Path to simulation YAML config")
    parser.add_argument("--headless", action="store_true", help="Run with SDL dummy video driver")
    parser.add_argument("--max-frames", type=int, default=900, help="Stop after N frames")
    parser.add_argument("--report", type=str, help="Write a JSON simulation report to this path")
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = resolve_repo_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


class TrafficSimulation:
    def __init__(self, config: dict, headless: bool = False):
        import pygame

        self.pygame = pygame
        self.config = config
        self.headless = headless
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        background_path = resolve_repo_path("assets/images/intersection.png")
        self.background = pygame.image.load(str(background_path))
        self.screen = pygame.display.set_mode(self.background.get_size())
        pygame.display.set_caption("Adaptive Traffic Simulation")
        self.font = pygame.font.Font(None, 28)
        self.clock = pygame.time.Clock()
        self.fps = int(config.get("window", {}).get("fps", 30))
        self.spawn_interval_frames = int(config.get("spawn", {}).get("interval_frames", 24))
        self.max_active_vehicles = int(config.get("spawn", {}).get("max_active_vehicles", 32))
        self.yellow_frames = int(config.get("signal", {}).get("yellow_time", 2) * self.fps)
        self.adaptive_timing = config.get("adaptive_timing", {})
        self.signal_images = {
            "red": pygame.image.load(str(resolve_repo_path("assets/images/signals/red.png"))),
            "yellow": pygame.image.load(str(resolve_repo_path("assets/images/signals/yellow.png"))),
            "green": pygame.image.load(str(resolve_repo_path("assets/images/signals/green.png"))),
        }
        self.vehicle_images = {
            direction: {
                vehicle_type: pygame.image.load(str(resolve_repo_path(f"assets/images/{direction}/{vehicle_type}.png")))
                for vehicle_type in VEHICLE_TYPES
            }
            for direction in DIRECTIONS
        }
        self.vehicles: list[Vehicle] = []
        self.frame_count = 0
        self.current_green_index = 0
        self.phase = "green"
        self.signal_switches = 0
        self.report = {
            "frames_processed": 0,
            "vehicles_spawned": 0,
            "vehicles_completed": 0,
            "signal_switches": 0,
            "queue_lengths": {direction: 0 for direction in DIRECTIONS},
            "adaptive_green_times": {},
            "warnings": [],
            "limitations": [
                "Simulation uses synthetic spawn rates, not live detector counts.",
                "Vehicle-to-vehicle collision avoidance and lane discipline are not modeled.",
                "Legacy four-way timing scripts remain for reference only.",
            ],
        }
        self.phase_frames_left = self._green_duration_frames(DIRECTIONS[self.current_green_index])

    def _green_duration_frames(self, active_direction: str) -> int:
        lane_traffic: dict[str, dict[str, list[object]]] = {direction: {} for direction in DIRECTIONS}
        for vehicle in self.vehicles:
            if vehicle.crossed:
                continue
            category = "motorbike" if vehicle.vehicle_type == "bike" else vehicle.vehicle_type
            lane_traffic[vehicle.direction].setdefault(category, []).append(object())

        green_times = calculate_adaptive_green_times(lane_traffic, self.adaptive_timing) or {}
        self.report["adaptive_green_times"] = green_times
        seconds = green_times.get(active_direction, self.config.get("signal", {}).get("base_green_time", 8))
        return max(1, int(seconds * self.fps))

    def _spawn_vehicle(self) -> None:
        if len(self.vehicles) >= self.max_active_vehicles:
            return
        direction = random.choice(DIRECTIONS)
        vehicle_type = random.choice(VEHICLE_TYPES)
        spawn_x, spawn_y = SPAWN_POSITIONS[direction]
        self.vehicles.append(Vehicle(vehicle_type, direction, spawn_x, spawn_y, SPEEDS[vehicle_type]))
        self.report["vehicles_spawned"] += 1

    def _can_move(self, vehicle: Vehicle) -> bool:
        green_direction = DIRECTIONS[self.current_green_index]
        if vehicle.crossed:
            return True

        if vehicle.direction == "right":
            front = vehicle.x + self.vehicle_images[vehicle.direction][vehicle.vehicle_type].get_width()
            if front >= STOP_LINES["right"]:
                vehicle.crossed = green_direction == "right" and self.phase == "green"
            return green_direction == "right" and self.phase == "green" or front < STOP_LINES["right"]
        if vehicle.direction == "down":
            front = vehicle.y + self.vehicle_images[vehicle.direction][vehicle.vehicle_type].get_height()
            if front >= STOP_LINES["down"]:
                vehicle.crossed = green_direction == "down" and self.phase == "green"
            return green_direction == "down" and self.phase == "green" or front < STOP_LINES["down"]
        if vehicle.direction == "left":
            front = vehicle.x
            if front <= STOP_LINES["left"]:
                vehicle.crossed = green_direction == "left" and self.phase == "green"
            return green_direction == "left" and self.phase == "green" or front > STOP_LINES["left"]
        front = vehicle.y
        if front <= STOP_LINES["up"]:
            vehicle.crossed = green_direction == "up" and self.phase == "green"
        return green_direction == "up" and self.phase == "green" or front > STOP_LINES["up"]

    def _move_vehicle(self, vehicle: Vehicle) -> bool:
        if self._can_move(vehicle):
            if vehicle.direction == "right":
                vehicle.x += vehicle.speed
            elif vehicle.direction == "down":
                vehicle.y += vehicle.speed
            elif vehicle.direction == "left":
                vehicle.x -= vehicle.speed
            else:
                vehicle.y -= vehicle.speed

        limit = EXIT_LIMITS[vehicle.direction]
        if vehicle.direction == "right" and vehicle.x > limit:
            return True
        if vehicle.direction == "down" and vehicle.y > limit:
            return True
        if vehicle.direction == "left" and vehicle.x < limit:
            return True
        if vehicle.direction == "up" and vehicle.y < limit:
            return True
        return False

    def _update_signal_phase(self) -> None:
        self.phase_frames_left -= 1
        if self.phase_frames_left > 0:
            return

        if self.phase == "green":
            self.phase = "yellow"
            self.phase_frames_left = self.yellow_frames
            return

        self.phase = "green"
        self.current_green_index = (self.current_green_index + 1) % len(DIRECTIONS)
        self.phase_frames_left = self._green_duration_frames(DIRECTIONS[self.current_green_index])
        self.signal_switches += 1
        self.report["signal_switches"] = self.signal_switches

    def _queue_lengths(self) -> dict[str, int]:
        queue_lengths = {direction: 0 for direction in DIRECTIONS}
        for vehicle in self.vehicles:
            if vehicle.crossed:
                continue
            direction = vehicle.direction
            axis = QUEUE_AXES[direction]
            sign = QUEUE_SIGNS[direction]
            coord = getattr(vehicle, axis)
            stop = STOP_LINES[direction]
            if sign == 1 and coord <= stop:
                queue_lengths[direction] += 1
            elif sign == -1 and coord >= stop:
                queue_lengths[direction] += 1
        return queue_lengths

    def _draw(self) -> None:
        self.screen.blit(self.background, (0, 0))
        active_direction = DIRECTIONS[self.current_green_index]
        for direction in DIRECTIONS:
            if direction == active_direction:
                signal_key = "yellow" if self.phase == "yellow" else "green"
            else:
                signal_key = "red"
            self.screen.blit(self.signal_images[signal_key], SIGNAL_POSITIONS[direction])
            seconds_left = round(self.phase_frames_left / self.fps, 1)
            text = self.font.render(str(seconds_left), True, (255, 255, 255))
            self.screen.blit(text, SIGNAL_TIMER_POSITIONS[direction])

        y = 20
        for direction, queue in self.report["queue_lengths"].items():
            label = self.font.render(f"{direction}: queue={queue}", True, (255, 255, 255))
            self.screen.blit(label, (20, y))
            y += 26

        for vehicle in self.vehicles:
            image = self.vehicle_images[vehicle.direction][vehicle.vehicle_type]
            self.screen.blit(image, (vehicle.x, vehicle.y))

        self.pygame.display.update()

    def run(self, max_frames: int) -> dict:
        while self.frame_count < max_frames:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.report["warnings"].append("Simulation stopped by window close.")
                    return self.report

            if self.frame_count % self.spawn_interval_frames == 0:
                self._spawn_vehicle()

            remaining: list[Vehicle] = []
            for vehicle in self.vehicles:
                finished = self._move_vehicle(vehicle)
                if finished:
                    self.report["vehicles_completed"] += 1
                else:
                    remaining.append(vehicle)
            self.vehicles = remaining

            self.report["queue_lengths"] = self._queue_lengths()
            self._update_signal_phase()
            self._draw()
            self.clock.tick(0 if self.headless else self.fps)
            self.frame_count += 1
            self.report["frames_processed"] = self.frame_count

        return self.report


def main() -> int:
    args = parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    config = load_config(args.config)
    simulation = TrafficSimulation(config, headless=args.headless)
    try:
        report = simulation.run(args.max_frames)
    finally:
        simulation.pygame.quit()

    if args.report:
        report_path = resolve_repo_path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
