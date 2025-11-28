"""
CHECKPOINT.PY - Tracks race progress
Checkpoints are invisible lines the car must cross in order
Prevents AI from cheating by going backwards
"""

import pygame
from scripts.Constants import TRACK_CHECKPOINT_ZONES, FONT


def lines_intersect(p1, p2, p3, p4):
    """Math to check if two line segments cross each other"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(denom) < 1e-10:  # Lines are parallel
        return False
    
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denom
    
    return 0 <= t <= 1 and 0 <= u <= 1


class CheckpointManager:
    def __init__(self):
        self.zones = TRACK_CHECKPOINT_ZONES  # List of checkpoint line coordinates
        self.total_checkpoints = len(self.zones)
        self.font = pygame.font.Font(FONT, 12)
        self.reset()

    def reset(self):
        self.current_idx = 0      # Next checkpoint to cross
        self.crossed_count = 0    # Total crossed
        self.cross_counts = [0] * self.total_checkpoints  # Times each was crossed
        self.prev_pos = None      # Previous car position

    def check_crossing(self, car_pos):
        """Check if car crossed any checkpoint. Returns (forward, backward)"""
        if self.prev_pos is None:
            self.prev_pos = car_pos
            return False, False

        # Check current checkpoint
        if self.current_idx < self.total_checkpoints:
            p1, p2 = self.zones[self.current_idx]
            if lines_intersect(self.prev_pos, car_pos, p1, p2):
                self.cross_counts[self.current_idx] += 1
                self.crossed_count += 1
                self.current_idx += 1
                self.prev_pos = car_pos
                return True, False

        # Check if crossed backwards through already-passed checkpoints
        for i in range(self.current_idx):
            p1, p2 = self.zones[i]
            if lines_intersect(self.prev_pos, car_pos, p1, p2):
                self.cross_counts[i] += 1
                self.prev_pos = car_pos
                return False, True

        self.prev_pos = car_pos
        return False, False

    def draw(self, surface):
        """Visualize checkpoints - green=next, gray=passed, red=crossed multiple times"""
        for i, (p1, p2) in enumerate(self.zones):
            if i == self.current_idx:
                color, width = (0, 255, 0), 4  # Green = current target
            elif i < self.current_idx:
                color = (255, 0, 0) if self.cross_counts[i] > 1 else (100, 100, 100)
                width = 3 if self.cross_counts[i] > 1 else 2
            else:
                color, width = (0, 100, 0), 2  # Dark green = upcoming
            
            pygame.draw.line(surface, color, p1, p2, width)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            pygame.draw.circle(surface, color, (cx, cy), 4)
            
            if self.cross_counts[i] > 0:
                text = self.font.render(f"x{self.cross_counts[i]}", True, (255, 255, 255))
                surface.blit(text, (cx + 10, cy - 10))