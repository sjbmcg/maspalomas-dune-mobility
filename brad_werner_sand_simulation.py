# Werner Dunes Model - cleaned up version
# Based on Werner (1995) - sand dunes via cellular automaton
# Maspalomas initial conditions with automatic frame snapshots

import pygame
import numpy as np
import random

class WernerDunesSimulation:
    
    def __init__(self, grid_width, grid_height, wind_from_deg=45, wind_speed_ms=10, 
                 initial_height_field=None):
        # Grid dimensions
        self.w = grid_width
        self.h = grid_height
        self.wind_from = wind_from_deg  # Wind FROM 45° (NE) blows TO 225° (SW)
        self.wind_speed = wind_speed_ms
        
        # Phase correlation analysis says dunes migrate 10m/year toward SW
        # Wind FROM northeast at 10 m/s transports sand southwestward
        self.hop_len = max(1, int(wind_speed_ms / 5))
        
        # Migration rate check
        self.target_migration_m_per_year = 10.0
        print(f"Target migration: {self.target_migration_m_per_year} m/year toward SW")
        print(f"Wind FROM NE (45°) at {wind_speed_ms} m/s")
        
        # Initialize height field
        if initial_height_field is not None:
            # Using actual Maspalomas initial conditions
            if initial_height_field.shape != (grid_height, grid_width):
                from scipy.ndimage import zoom
                zy = grid_height / initial_height_field.shape[0]
                zx = grid_width / initial_height_field.shape[1]
                self.z = zoom(initial_height_field, (zy, zx), order=1).astype(np.int32)
                print(f"Resized initial field from {initial_height_field.shape} to ({grid_height}, {grid_width})")
            else:
                self.z = initial_height_field.astype(np.int32)
            print(f"Loaded Maspalomas IC: mean={np.mean(self.z):.2f}, std={np.std(self.z):.2f}, range={self.z.min()}-{self.z.max()}")
        else:
            # Start with flat desert at height 5
            self.z = np.full((grid_height, grid_width), 5, dtype=np.int32)
            print("Using flat initial conditions")
        
        # PyGame setup
        pygame.init()
        self.cell_sz = min(6, 800 // max(grid_width, grid_height))
        screen_w = grid_width * self.cell_sz
        screen_h = grid_height * self.cell_sz
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption(f"Werner Dunes - Wind FROM NE: {wind_speed_ms} m/s")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.frame = 0
        
        # Recursion limiter for avalanches this just crashed my PC once
        self.rec_depth = 0
        self.max_rec = 100
    
    def wrap_x(self, x):
        return x % self.w
    
    def wrap_y(self, y):
        return y % self.h
    
    def try_avalanche(self, top_x, top_y, bot_x, bot_y):
        # Angle of repose: if height difference >= 3 cells, avalanche happens
        
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
        self.z[y, x] += 1
        
        # Check all neighbors
        self.try_avalanche(x, y, x+1, y)
        self.try_avalanche(x, y, x-1, y)
        self.try_avalanche(x, y, x, y-1)
        self.try_avalanche(x, y, x, y+1)
    
    def take(self, x, y):
        self.z[y, x] -= 1
        
        # Check neighbors
        self.try_avalanche(x-1, y, x, y)
        self.try_avalanche(x+1, y, x, y)
        self.try_avalanche(x, y-1, x, y)
        self.try_avalanche(x, y+1, x, y)
    
    def shadowed(self, x, y):
        # Wind shadow = lee side of dune where wind can't pick up sand
        # Wind FROM NE (45°) means scan toward NE to find obstacles
        
        dist = 1
        
        while dist < self.w:
            upwind_x = self.wrap_x(x + dist)  # east
            upwind_y = self.wrap_y(y + dist)  # north
            
            # Check east direction
            dh_east = self.z[y, upwind_x] - self.z[y, x]
            if dh_east >= dist:
                return True
            
            # Check north direction  
            dh_north = self.z[upwind_y, x] - self.z[y, x]
            if dh_north >= dist:
                return True
                
            dist += 1
        
        return False
    
    def blow(self):
        # One saltation event: pick up grain, blow it downwind toward SW
        
        tries = 0
        while tries < 1000:
            rx = random.randint(0, self.w - 1)
            ry = random.randint(0, self.h - 1)
            
            has_sand = self.z[ry, rx] != 0
            exposed = not self.shadowed(rx, ry)
            
            if has_sand and exposed:
                break
            tries += 1
        
        if tries >= 1000:
            return
        
        # Erode grain from source
        self.take(rx, ry)
        
        # Deposit grain downwind toward SW
        land_x = self.wrap_x(rx - self.hop_len)  # west component
        land_y = self.wrap_y(ry - 1)  # south component
        self.drop(land_x, land_y)
    
    def save_snapshot(self, frame_num):
        """Save height field snapshot as PNG"""
        filename = f"dune_state_frame{frame_num}.png"
        pygame.image.save(self.screen, filename)
        print(f"Saved snapshot: {filename}")
    
    def run(self):
        running = True
        paused = False
        
        print(f"Running Werner model with {self.wind_speed} m/s wind FROM NE...")
        print(f"Saltation length: {self.hop_len} cells toward SW")
        print("Press SPACE to pause, ESC to quit, S to save frame")
        print("Auto-saving frames 1 and 100...")
        
        while running:
            # Event handling
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
                # Do 1000 saltation events per frame
                for _ in range(1000):
                    self.blow()
                
                # Render the desert as a heatmap
                max_h = max(np.max(self.z), 10)
                brightness = self.z / max_h
                brightness = np.clip(brightness, 0, 1)
                
                # RGB image - sandy color palette
                img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img[:,:,0] = (255 * brightness).astype(np.uint8)
                img[:,:,1] = (230 * brightness).astype(np.uint8)
                img[:,:,2] = (160 * brightness).astype(np.uint8)
                
                # Scale up if cells are bigger than 1 pixel
                if self.cell_sz > 1:
                    img = np.repeat(np.repeat(img, self.cell_sz, axis=0), 
                                   self.cell_sz, axis=1)
                
                # Blit to screen
                surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
                
                # Stats overlay
                stats = [
                    f"Frame: {self.frame}",
                    f"Mean h: {np.mean(self.z):.1f}",
                    f"Std: {np.std(self.z):.2f}",
                    f"Range: {self.z.min()}-{self.z.max()}",
                    f"Wind FROM NE: {self.wind_speed} m/s"
                ]
                
                for i, txt in enumerate(stats):
                    label = self.font.render(txt, True, (255, 255, 255))
                    self.screen.blit(label, (10, 10 + i * 25))
                
                pygame.display.flip()
                self.frame += 1
                
                # Auto-save frames 1 and 100
                if self.frame == 1 or self.frame == 100:
                    self.save_snapshot(self.frame)
                
                # Console logging every 100 frames
                if self.frame % 100 == 0:
                    print(f"Frame {self.frame} | "
                          f"Std: {np.std(self.z):.3f} | "
                          f"Range: {self.z.min()}-{self.z.max()}")
            
            self.clock.tick(60)
        
        pygame.quit()
        print("Simulation ended. Check output CSVs.")


if __name__ == "__main__":
    maspalomas_ic = np.loadtxt("data/maspalomas_simulation_config/2020_continuous_topoIC.csv", delimiter=",").astype(np.int32)
    print("Loaded Maspalomas IC file.")
    
    sim = WernerDunesSimulation(200, 200, wind_from_deg=45, wind_speed_ms=10,
                               initial_height_field=maspalomas_ic)
    sim.run()