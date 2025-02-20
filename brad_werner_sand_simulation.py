"""Werner-style dune simulation with configurable, reproducible runs."""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pygame


class WernerDunesSimulation:
    """Simple cellular-automaton dune model with optional snapshot output."""

    def __init__(
        self,
        grid_width,
        grid_height,
        wind_from_deg=45,
        wind_speed_ms=10,
        initial_height_field=None,
        *,
        events_per_frame=1000,
        max_frames=None,
        snapshot_frames=None,
        output_dir=".",
        seed=42,
        headless=False,
    ):
        self.w = grid_width
        self.h = grid_height
        self.wind_from = wind_from_deg % 360
        self.wind_to = (self.wind_from + 180) % 360
        self.wind_speed = wind_speed_ms
        self.events_per_frame = events_per_frame
        self.max_frames = max_frames
        self.snapshot_frames = set(snapshot_frames or [1, 100])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = random.Random(seed)

        self.upwind_dx, self.upwind_dy = self._grid_vector(self.wind_from)
        self.downwind_dx, self.downwind_dy = self._grid_vector(self.wind_to)

        # This remains a heuristic mapping between wind speed and hop length.
        self.hop_len = max(1, int(wind_speed_ms / 5))
        self.target_migration_m_per_year = 10.0

        if initial_height_field is not None:
            if initial_height_field.shape != (grid_height, grid_width):
                from scipy.ndimage import zoom

                zy = grid_height / initial_height_field.shape[0]
                zx = grid_width / initial_height_field.shape[1]
                self.z = zoom(initial_height_field, (zy, zx), order=1).astype(np.int32)
                print(
                    f"Resized initial field from {initial_height_field.shape} "
                    f"to ({grid_height}, {grid_width})"
                )
            else:
                self.z = initial_height_field.astype(np.int32)
            print(
                "Loaded Maspalomas IC: "
                f"mean={np.mean(self.z):.2f}, std={np.std(self.z):.2f}, "
                f"range={self.z.min()}-{self.z.max()}"
            )
        else:
            self.z = np.full((grid_height, grid_width), 5, dtype=np.int32)
            print("Using flat initial conditions")

        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        self.cell_sz = max(1, min(6, 800 // max(grid_width, grid_height)))
        screen_w = grid_width * self.cell_sz
        screen_h = grid_height * self.cell_sz
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption(
            f"Werner Dunes - wind from {self.wind_from:.0f} deg at {wind_speed_ms} m/s"
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.frame = 0
        self.rec_depth = 0
        self.max_rec = 100

        print(
            f"Target migration: about {self.target_migration_m_per_year} m/year downwind"
        )
        print(
            f"Wind from {self.wind_from:.0f} degrees toward {self.wind_to:.0f} degrees "
            f"at {wind_speed_ms} m/s"
        )
        print(f"Random seed: {self.seed}")

    @staticmethod
    def _grid_vector(direction_deg):
        """Convert a compass bearing into a unit vector in grid coordinates."""
        radians = math.radians(direction_deg)
        dx = math.sin(radians)
        dy = -math.cos(radians)
        return dx, dy

    def wrap_x(self, x):
        return x % self.w

    def wrap_y(self, y):
        return y % self.h

    def try_avalanche(self, top_x, top_y, bot_x, bot_y):
        """Move one grain downslope if the local slope exceeds the threshold."""
        if self.rec_depth >= self.max_rec:
            return

        top_x = self.wrap_x(top_x)
        bot_x = self.wrap_x(bot_x)
        top_y = self.wrap_y(top_y)
        bot_y = self.wrap_y(bot_y)

        dh = self.z[top_y, top_x] - self.z[bot_y, bot_x]
        if dh >= 3:
            self.rec_depth += 1
            self.take(top_x, top_y)
            self.drop(bot_x, bot_y)
            self.rec_depth -= 1

    def drop(self, x, y):
        x = self.wrap_x(x)
        y = self.wrap_y(y)
        self.z[y, x] += 1

        self.try_avalanche(x, y, x + 1, y)
        self.try_avalanche(x, y, x - 1, y)
        self.try_avalanche(x, y, x, y - 1)
        self.try_avalanche(x, y, x, y + 1)

    def take(self, x, y):
        x = self.wrap_x(x)
        y = self.wrap_y(y)
        if self.z[y, x] <= 0:
            return

        self.z[y, x] -= 1
        self.try_avalanche(x - 1, y, x, y)
        self.try_avalanche(x + 1, y, x, y)
        self.try_avalanche(x, y - 1, x, y)
        self.try_avalanche(x, y + 1, x, y)

    def shadowed(self, x, y):
        """Return True when a cell lies in the upwind shadow of higher terrain."""
        source_height = self.z[y, x]
        max_dist = max(self.w, self.h)

        for dist in range(1, max_dist):
            upwind_x = self.wrap_x(int(round(x + dist * self.upwind_dx)))
            upwind_y = self.wrap_y(int(round(y + dist * self.upwind_dy)))
            dh = self.z[upwind_y, upwind_x] - source_height
            if dh >= dist:
                return True

        return False

    def blow(self):
        """Perform one saltation event."""
        tries = 0
        while tries < 1000:
            rx = self.rng.randint(0, self.w - 1)
            ry = self.rng.randint(0, self.h - 1)

            has_sand = self.z[ry, rx] > 0
            exposed = not self.shadowed(rx, ry)
            if has_sand and exposed:
                break
            tries += 1

        if tries >= 1000:
            return

        self.take(rx, ry)
        land_x = self.wrap_x(int(round(rx + self.hop_len * self.downwind_dx)))
        land_y = self.wrap_y(int(round(ry + self.hop_len * self.downwind_dy)))
        self.drop(land_x, land_y)

    def render_frame(self):
        """Draw the current state to the pygame surface."""
        max_h = max(np.max(self.z), 10)
        brightness = np.clip(self.z / max_h, 0, 1)

        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:, :, 0] = (255 * brightness).astype(np.uint8)
        img[:, :, 1] = (230 * brightness).astype(np.uint8)
        img[:, :, 2] = (160 * brightness).astype(np.uint8)

        if self.cell_sz > 1:
            img = np.repeat(np.repeat(img, self.cell_sz, axis=0), self.cell_sz, axis=1)

        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        self.screen.blit(surf, (0, 0))

        stats = [
            f"Frame: {self.frame}",
            f"Mean h: {np.mean(self.z):.1f}",
            f"Std: {np.std(self.z):.2f}",
            f"Range: {self.z.min()}-{self.z.max()}",
            f"Wind: {self.wind_from:.0f} deg at {self.wind_speed} m/s",
        ]
        for i, txt in enumerate(stats):
            label = self.font.render(txt, True, (255, 255, 255))
            self.screen.blit(label, (10, 10 + i * 25))

        pygame.display.flip()

    def save_snapshot(self, frame_num):
        """Save the rendered frame as a PNG in the configured output directory."""
        filename = self.output_dir / f"dune_state_frame{frame_num}.png"
        pygame.image.save(self.screen, str(filename))
        print(f"Saved snapshot: {filename}")

    def run(self):
        """Run the simulation loop."""
        running = True
        paused = False

        print(
            f"Running Werner model with {self.wind_speed} m/s wind "
            f"from {self.wind_from:.0f} degrees"
        )
        print(f"Saltation length: {self.hop_len} cells downwind")
        print(f"Events per frame: {self.events_per_frame}")
        if self.max_frames is not None:
            print(f"Stopping automatically after {self.max_frames} frames")
        if self.snapshot_frames:
            print(f"Snapshot frames: {sorted(self.snapshot_frames)}")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_s:
                        self.save_snapshot(self.frame)

            if not paused:
                for _ in range(self.events_per_frame):
                    self.blow()

                self.render_frame()
                self.frame += 1

                if self.frame in self.snapshot_frames:
                    self.save_snapshot(self.frame)

                if self.frame % 100 == 0:
                    print(
                        f"Frame {self.frame} | "
                        f"Std: {np.std(self.z):.3f} | "
                        f"Range: {self.z.min()}-{self.z.max()}"
                    )

                if self.max_frames is not None and self.frame >= self.max_frames:
                    running = False

            self.clock.tick(60)

        pygame.quit()
        print("Simulation ended. Check saved PNG snapshots if you enabled them.")


def load_initial_conditions(path):
    """Load a CSV height field from disk."""
    initial_conditions = np.loadtxt(path, delimiter=",").astype(np.int32)
    print(f"Loaded initial-condition file: {path}")
    return initial_conditions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-conditions",
        default="data/maspalomas_simulation_config/2020_continuous_topoIC.csv",
        help="CSV file containing the initial height field.",
    )
    parser.add_argument("--grid-width", type=int, default=200)
    parser.add_argument("--grid-height", type=int, default=200)
    parser.add_argument("--wind-from-deg", type=float, default=45)
    parser.add_argument("--wind-speed-ms", type=float, default=10)
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


def main():
    args = parse_args()
    initial_height_field = load_initial_conditions(args.initial_conditions)
    sim = WernerDunesSimulation(
        args.grid_width,
        args.grid_height,
        wind_from_deg=args.wind_from_deg,
        wind_speed_ms=args.wind_speed_ms,
        initial_height_field=initial_height_field,
        events_per_frame=args.events_per_frame,
        max_frames=args.max_frames,
        snapshot_frames=args.snapshot_frames,
        output_dir=args.output_dir,
        seed=args.seed,
        headless=args.headless,
    )
    sim.run()


if __name__ == "__main__":
    main()
