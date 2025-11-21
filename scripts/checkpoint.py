# scripts/checkpoint.py - SIMPLIFIED (No Shapely dependency)
import pygame
from scripts.Constants import TRACK_CHECKPOINT_ZONES, FONT


def line_intersects(p1, p2, p3, p4):
    """
    Fast line-line intersection check (no library needed)
    Returns True if line segment (p1,p2) intersects (p3,p4)
    Uses cross product method - faster than Shapely for single checks
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Calculate direction vectors
    d1x = x2 - x1
    d1y = y2 - y1
    d2x = x4 - x3
    d2y = y4 - y3
    
    # Calculate cross products
    denom = d1x * d2y - d1y * d2x
    
    # Lines are parallel if denominator is 0
    if abs(denom) < 1e-10:
        return False
    
    # Calculate intersection parameters
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
    u = ((x3 - x1) * d1y - (y3 - y1) * d1x) / denom
    
    # Check if intersection point is within both line segments
    return 0 <= t <= 1 and 0 <= u <= 1


class CheckpointManager:
    def __init__(self):
        self.zones = TRACK_CHECKPOINT_ZONES
        self.total_checkpoints = len(TRACK_CHECKPOINT_ZONES)
        self.font = pygame.font.Font(FONT, 12)
        self.reset()

    def check_crossing(self, car_pos):
        """Check if car crossed any checkpoint"""
        if self.prev_car_pos is None:
            self.prev_car_pos = car_pos
            return False, False

        if self.current_idx >= self.total_checkpoints:
            self.prev_car_pos = car_pos
            return False, False

        # Check current checkpoint
        cp_p1, cp_p2 = self.zones[self.current_idx]
        if line_intersects(self.prev_car_pos, car_pos, cp_p1, cp_p2):
            self.checkpoint_cross_counts[self.current_idx] += 1
            self.crossed_count += 1
            self.current_idx += 1
            self.prev_car_pos = car_pos
            return True, False

        # Check backward crossings (only for completed checkpoints)
        for i in range(self.current_idx):
            cp_p1, cp_p2 = self.zones[i]
            if line_intersects(self.prev_car_pos, car_pos, cp_p1, cp_p2):
                self.checkpoint_cross_counts[i] += 1
                self.prev_car_pos = car_pos
                return False, True

        self.prev_car_pos = car_pos
        return False, False

    def reset(self):
        self.current_idx = 0
        self.crossed_count = 0
        self.checkpoint_cross_counts = [0] * self.total_checkpoints
        self.prev_car_pos = None

    def draw(self, surface):       
        for i, (p1, p2) in enumerate(self.zones):
            # Color based on state
            if i == self.current_idx:
                color, width = (0, 255, 0), 4  # Current checkpoint
            elif i < self.current_idx:
                # Completed - red if crossed multiple times, else gray
                color = (255, 0, 0) if self.checkpoint_cross_counts[i] > 1 else (100, 100, 100)
                width = 3 if self.checkpoint_cross_counts[i] > 1 else 2
            else:
                color, width = (0, 100, 0), 2  # Upcoming
            
            # Draw checkpoint line and center dot
            pygame.draw.line(surface, color, p1, p2, width)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            pygame.draw.circle(surface, color, (cx, cy), 4)
            
            # Draw cross count if > 0
            if self.checkpoint_cross_counts[i] > 0:
                text = self.font.render(f"x{self.checkpoint_cross_counts[i]}", True, (255, 255, 255))
                # Simple black background
                bg = pygame.Surface((text.get_width() + 4, text.get_height() + 2))
                bg.set_alpha(180)
                bg.fill((0, 0, 0))
                surface.blit(bg, (cx + 10, cy - 10))
                surface.blit(text, (cx + 12, cy - 9))