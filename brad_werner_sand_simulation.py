"""Werner-style dune simulation with configurable, reproducible runs."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame


AVALANCHE_THRESHOLD = 3
MAX_SOURCE_SELECTION_TRIES = 1000


@dataclass(frozen=True)
class SimulationConfig:
    grid_width: int
    grid_height: int
    wind_from_deg: float = 45.0
    wind_speed_ms: float = 10.0
    events_per_frame: int = 1000
    max_frames: int | None = None
    snapshot_frames: tuple[int, ...] = (1, 100)
    output_dir: Path = Path(".")
    seed: int = 42
    headless: bool = False

    @property
    def wind_from(self) -> float:
        return self.wind_from_deg % 360

    @property
    def wind_to(self) -> float:
        return (self.wind_from + 180.0) % 360

    @property
    def hop_length(self) -> int:
        # This remains a heuristic mapping between wind speed and hop length.
        return max(1, int(self.wind_speed_ms / 5))


def compass_to_grid_vector(direction_deg: float) -> tuple[float, float]:
    """Convert a compass bearing into a unit vector in grid coordinates."""
    radians = math.radians(direction_deg)
    dx = math.sin(radians)
    dy = -math.cos(radians)
    return dx, dy


def resize_height_field(
    height_field: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Resize a height field to the model grid if the shapes differ."""
    if height_field.shape == (target_height, target_width):
        return height_field.astype(np.int32)

    from scipy.ndimage import zoom

    zoom_y = target_height / height_field.shape[0]
    zoom_x = target_width / height_field.shape[1]
    resized = zoom(height_field, (zoom_y, zoom_x), order=1)
    return resized.astype(np.int32)


def load_initial_conditions(path: str | Path) -> np.ndarray:
    """Load a CSV height field from disk."""
    file_path = Path(path)
    initial_conditions = np.loadtxt(file_path, delimiter=",").astype(np.int32)
    print(f"Loaded initial-condition file: {file_path}")
    return initial_conditions


class DuneField:
    """Cellular dune field with saltation and local slope relaxation."""

    def __init__(self, config: SimulationConfig, initial_height_field: np.ndarray | None):
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height
        self.rng = random.Random(config.seed)
        self.upwind_dx, self.upwind_dy = compass_to_grid_vector(config.wind_from)
        self.downwind_dx, self.downwind_dy = compass_to_grid_vector(config.wind_to)
        self.heights = self._initialize_heights(initial_height_field)

    def _initialize_heights(self, initial_height_field: np.ndarray | None) -> np.ndarray:
        if initial_height_field is None:
            print("Using flat initial conditions")
            return np.full((self.height, self.width), 5, dtype=np.int32)

        resized = resize_height_field(initial_height_field, self.height, self.width)
        if initial_height_field.shape != resized.shape:
            print(
                f"Resized initial field from {initial_height_field.shape} "
                f"to ({self.height}, {self.width})"
            )

        print(
            "Loaded Maspalomas IC: "
            f"mean={np.mean(resized):.2f}, std={np.std(resized):.2f}, "
            f"range={resized.min()}-{resized.max()}"
        )
        return resized

    def wrap_x(self, x: int) -> int:
        return x % self.width

    def wrap_y(self, y: int) -> int:
        return y % self.height

    def _wrapped_point(self, x: int, y: int) -> tuple[int, int]:
        return self.wrap_x(x), self.wrap_y(y)

    def _neighbor_points(self, x: int, y: int) -> tuple[tuple[int, int], ...]:
        return (
            (x + 1, y),
            (x - 1, y),
            (x, y - 1),
            (x, y + 1),
        )

    def _enqueue_drop_relaxation(
        self, queue: deque[tuple[tuple[int, int], tuple[int, int]]], x: int, y: int
    ) -> None:
        for neighbor_x, neighbor_y in self._neighbor_points(x, y):
            queue.append(((x, y), (neighbor_x, neighbor_y)))

    def _enqueue_take_relaxation(
        self, queue: deque[tuple[tuple[int, int], tuple[int, int]]], x: int, y: int
    ) -> None:
        for neighbor_x, neighbor_y in self._neighbor_points(x, y):
            queue.append(((neighbor_x, neighbor_y), (x, y)))

    def _relax_queue(
        self, queue: deque[tuple[tuple[int, int], tuple[int, int]]]
    ) -> None:
        while queue:
            (top_x, top_y), (bottom_x, bottom_y) = queue.popleft()
            top_x, top_y = self._wrapped_point(top_x, top_y)
            bottom_x, bottom_y = self._wrapped_point(bottom_x, bottom_y)

            dh = self.heights[top_y, top_x] - self.heights[bottom_y, bottom_x]
            if dh < AVALANCHE_THRESHOLD:
                continue

            self.heights[top_y, top_x] -= 1
            self.heights[bottom_y, bottom_x] += 1

            self._enqueue_take_relaxation(queue, top_x, top_y)
            self._enqueue_drop_relaxation(queue, bottom_x, bottom_y)

    def _remove_grain(self, x: int, y: int) -> bool:
        x, y = self._wrapped_point(x, y)
        if self.heights[y, x] <= 0:
            return False

        self.heights[y, x] -= 1
        queue: deque[tuple[tuple[int, int], tuple[int, int]]] = deque()
        self._enqueue_take_relaxation(queue, x, y)
        self._relax_queue(queue)
        return True

    def _add_grain(self, x: int, y: int) -> None:
        x, y = self._wrapped_point(x, y)
        self.heights[y, x] += 1
        queue: deque[tuple[tuple[int, int], tuple[int, int]]] = deque()
        self._enqueue_drop_relaxation(queue, x, y)
        self._relax_queue(queue)

    def is_shadowed(self, x: int, y: int) -> bool:
        """Return True when a cell lies in the upwind shadow of higher terrain."""
        source_height = self.heights[y, x]
        max_dist = max(self.width, self.height)

        for dist in range(1, max_dist):
            upwind_x = self.wrap_x(int(round(x + dist * self.upwind_dx)))
            upwind_y = self.wrap_y(int(round(y + dist * self.upwind_dy)))
            height_difference = self.heights[upwind_y, upwind_x] - source_height
            if height_difference >= dist:
                return True

        return False

    def _choose_exposed_source(self) -> tuple[int, int] | None:
        for _ in range(MAX_SOURCE_SELECTION_TRIES):
            x = self.rng.randint(0, self.width - 1)
            y = self.rng.randint(0, self.height - 1)
            if self.heights[y, x] > 0 and not self.is_shadowed(x, y):
                return x, y
        return None

    def blow_one_event(self) -> None:
        """Perform one saltation event."""
        source = self._choose_exposed_source()
        if source is None:
            return

        source_x, source_y = source
        if not self._remove_grain(source_x, source_y):
            return

        land_x = self.wrap_x(
            int(round(source_x + self.config.hop_length * self.downwind_dx))
        )
        land_y = self.wrap_y(
            int(round(source_y + self.config.hop_length * self.downwind_dy))
        )
        self._add_grain(land_x, land_y)

    def run_transport_step(self) -> None:
        for _ in range(self.config.events_per_frame):
            self.blow_one_event()

    def summary_stats(self) -> dict[str, float | int]:
        return {
            "mean": float(np.mean(self.heights)),
            "std": float(np.std(self.heights)),
            "min": int(self.heights.min()),
            "max": int(self.heights.max()),
        }


class PygameView:
    """Thin pygame wrapper for rendering and interactive controls."""

    def __init__(self, config: SimulationConfig):
        if config.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        self.config = config
        self.cell_size = max(1, min(6, 800 // max(config.grid_width, config.grid_height)))
        screen_width = config.grid_width * self.cell_size
        screen_height = config.grid_height * self.cell_size
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(
            f"Werner Dunes - wind from {config.wind_from:.0f} deg at {config.wind_speed_ms} m/s"
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def poll(self) -> dict[str, bool]:
        controls = {"quit": False, "toggle_pause": False, "save_snapshot": False}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                controls["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    controls["quit"] = True
                elif event.key == pygame.K_SPACE:
                    controls["toggle_pause"] = True
                elif event.key == pygame.K_s:
                    controls["save_snapshot"] = True
        return controls

    def render(self, dune_field: DuneField, frame: int) -> None:
        """Draw the current state to the pygame surface."""
        max_height = max(int(np.max(dune_field.heights)), 10)
        brightness = np.clip(dune_field.heights / max_height, 0, 1)

        image = np.zeros((dune_field.height, dune_field.width, 3), dtype=np.uint8)
        image[:, :, 0] = (255 * brightness).astype(np.uint8)
        image[:, :, 1] = (230 * brightness).astype(np.uint8)
        image[:, :, 2] = (160 * brightness).astype(np.uint8)

        if self.cell_size > 1:
            image = np.repeat(np.repeat(image, self.cell_size, axis=0), self.cell_size, axis=1)

        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

        stats = dune_field.summary_stats()
        overlay_lines = [
            f"Frame: {frame}",
            f"Mean h: {stats['mean']:.1f}",
            f"Std: {stats['std']:.2f}",
            f"Range: {stats['min']}-{stats['max']}",
            f"Wind: {self.config.wind_from:.0f} deg at {self.config.wind_speed_ms:g} m/s",
        ]
        for index, text in enumerate(overlay_lines):
            label = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(label, (10, 10 + index * 25))

        pygame.display.flip()

    def save_snapshot(self, output_dir: Path, frame: int) -> Path:
        output_path = output_dir / f"dune_state_frame{frame}.png"
        pygame.image.save(self.screen, str(output_path))
        return output_path

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    @staticmethod
    def close() -> None:
        pygame.quit()


class SimulationRunner:
    """Coordinate the model, renderer, and runtime controls."""

    def __init__(self, config: SimulationConfig, initial_height_field: np.ndarray | None):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dune_field = DuneField(config, initial_height_field)
        self.view = PygameView(config)
        self.frame = 0

    def _print_startup_summary(self) -> None:
        stats = self.dune_field.summary_stats()
        print(
            f"Wind from {self.config.wind_from:.0f} deg toward {self.config.wind_to:.0f} deg "
            f"at {self.config.wind_speed_ms:g} m/s"
        )
        print(f"Saltation length: {self.config.hop_length} cells downwind")
        print(f"Events per frame: {self.config.events_per_frame}")
        print(f"Random seed: {self.config.seed}")
        print(
            "Initial field stats | "
            f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"range={stats['min']}-{stats['max']}"
        )
        if self.config.max_frames is not None:
            print(f"Stopping automatically after {self.config.max_frames} frames")
        if self.config.snapshot_frames:
            print(f"Snapshot frames: {list(self.config.snapshot_frames)}")

    def _save_snapshot(self) -> None:
        output_path = self.view.save_snapshot(self.output_dir, self.frame)
        print(f"Saved snapshot: {output_path}")

    def run(self) -> None:
        """Run the simulation loop."""
        self._print_startup_summary()
        running = True
        paused = False

        while running:
            controls = self.view.poll()
            if controls["quit"]:
                running = False
            if controls["toggle_pause"]:
                paused = not paused
            if controls["save_snapshot"]:
                self._save_snapshot()

            if not paused and running:
                self.dune_field.run_transport_step()
                self.view.render(self.dune_field, self.frame)
                self.frame += 1

                if self.frame in self.config.snapshot_frames:
                    self._save_snapshot()

                if self.frame % 100 == 0:
                    stats = self.dune_field.summary_stats()
                    print(
                        f"Frame {self.frame} | "
                        f"Std: {stats['std']:.3f} | "
                        f"Range: {stats['min']}-{stats['max']}"
                    )

                if self.config.max_frames is not None and self.frame >= self.config.max_frames:
                    running = False

            self.view.tick(60)

        self.view.close()
        print("Simulation ended. Check saved PNG snapshots if you enabled them.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-conditions",
        default="data/maspalomas_simulation_config/2020_continuous_topoIC.csv",
        help="CSV file containing the initial height field.",
    )
    parser.add_argument("--grid-width", type=int, default=200)
    parser.add_argument("--grid-height", type=int, default=200)
    parser.add_argument("--wind-from-deg", type=float, default=45.0)
    parser.add_argument("--wind-speed-ms", type=float, default=10.0)
    parser.add_argument("--events-per-frame", type=int, default=1000)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard stop for non-interactive runs.",
    )
    parser.add_argument(
        "--snapshot-frames",
        type=int,
        nargs="*",
        default=[1, 100],
        help="Frames to save as PNG snapshots.",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use the SDL dummy video driver for non-interactive runs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        wind_from_deg=args.wind_from_deg,
        wind_speed_ms=args.wind_speed_ms,
        events_per_frame=args.events_per_frame,
        max_frames=args.max_frames,
        snapshot_frames=tuple(args.snapshot_frames),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        headless=args.headless,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    initial_height_field = load_initial_conditions(args.initial_conditions)
    runner = SimulationRunner(config, initial_height_field)
    runner.run()


if __name__ == "__main__":
    main()
