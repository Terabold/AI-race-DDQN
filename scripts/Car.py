"""
CAR.PY - The racing car with physics and sensors
Handles movement, rotation, and "vision" via raycasting
Raycasting = shooting invisible lines to detect walls/obstacles (like sonar)
"""

import math
import pygame
import numpy as np
from pygame.math import Vector2
from scripts.Constants import CAR_COLORS, MAXSPEED, ROTATESPEED, ACCELERATION, GREEN, YELLOW

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y, car_color="Red"):
        super().__init__()
        self.position = Vector2(x, y)
        self.car_color = car_color

        # Load car image
        self.img = pygame.image.load(CAR_COLORS[car_color]).convert_alpha()
        self.image = pygame.transform.scale(self.img, (19, 38))
        self.original_image = self.image
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)  # For collision detection

        # Physics properties
        self.max_velocity = MAXSPEED
        self.velocity = 0
        self.rotation_velocity = ROTATESPEED
        self.angle = 0
        self.acceleration = ACCELERATION

        # State flags
        self.failed = False      # Crashed into wall
        self.can_move = True     # Can still move

        # RAYCAST SYSTEM - Two sets of "sensors"
        # Wall rays: Detect only walls (for navigation)
        # Bomb rays: Detect walls AND obstacles (for avoidance)
        self.ray_length = 400
        
        # 15 rays spread from -90° to +90° (left to right)
        self.wall_ray_angles = np.array([-90, -60, -45, -30, -20, -15, -10, 0, 10, 15, 20, 30, 45, 60, 90], dtype=np.float32)
        self.bomb_ray_angles = np.array([-90, -60, -45, -30, -20, -15, -10, 0, 10, 15, 20, 30, 45, 60, 90], dtype=np.float32)
        
        # Distance readings (normalized 0-1, where 1 = max distance = safe)
        self.wall_distances = np.full(len(self.wall_ray_angles), self.ray_length, dtype=np.float32)
        self.bomb_distances = np.full(len(self.bomb_ray_angles), self.ray_length, dtype=np.float32)
        self.bomb_hit_obstacle = np.zeros(len(self.bomb_ray_angles), dtype=bool)
        
        # For visualization
        self.wall_collision_points = [None] * len(self.wall_ray_angles)
        self.bomb_collision_points = [None] * len(self.bomb_ray_angles)
        
        # Pre-calculate ray directions for performance
        self.wall_directions = np.array([
            [math.sin(math.radians(-angle)), -math.cos(math.radians(-angle))]
            for angle in self.wall_ray_angles
        ], dtype=np.float32)
        
        self.bomb_directions = np.array([
            [math.sin(math.radians(-angle)), -math.cos(math.radians(-angle))]
            for angle in self.bomb_ray_angles
        ], dtype=np.float32)

    def cast_rays(self, border_mask, obstacle_group=None):
        """Shoot rays to detect surroundings - called every frame"""
        car_rotation = -self.angle
        step = 15  # Check every 10 pixels along ray
        width, height = border_mask.get_size()
        
        self.cast_wall_rays(border_mask, car_rotation, step, width, height)
        if obstacle_group:
            self.cast_bomb_rays(border_mask, obstacle_group, car_rotation, step, width, height)
    
    def cast_wall_rays(self, border_mask, car_rotation, step, width, height):
        """Detect walls only"""
        self.wall_distances.fill(self.ray_length)
        
        angle_rad = math.radians(car_rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        for idx, base_dir in enumerate(self.wall_directions):
            # Rotate ray direction by car angle
            ray_dir_x = base_dir[0] * cos_a - base_dir[1] * sin_a
            ray_dir_y = base_dir[0] * sin_a + base_dir[1] * cos_a
            
            min_dist = self.ray_length
            collision_point = None
            
            # March along ray until we hit something
            for dist in range(step, self.ray_length + 1, step):
                x = int(self.position.x + ray_dir_x * dist)
                y = int(self.position.y + ray_dir_y * dist)
                
                if not (0 <= x < width and 0 <= y < height):
                    break
                
                if border_mask.get_at((x, y)):  # Hit wall
                    min_dist = dist
                    collision_point = Vector2(x, y)
                    break
            
            self.wall_distances[idx] = min_dist
            self.wall_collision_points[idx] = collision_point
    
    def cast_bomb_rays(self, border_mask, obstacle_group, car_rotation, step, width, height):
        """Detect walls AND obstacles"""
        self.bomb_distances.fill(self.ray_length)
        self.bomb_hit_obstacle.fill(False)
        
        angle_rad = math.radians(car_rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        for idx, base_dir in enumerate(self.bomb_directions):
            ray_dir_x = base_dir[0] * cos_a - base_dir[1] * sin_a
            ray_dir_y = base_dir[0] * sin_a + base_dir[1] * cos_a
            
            min_dist = self.ray_length
            collision_point = None
            hit_obstacle = False
            
            for dist in range(step, self.ray_length + 1, step):
                x = int(self.position.x + ray_dir_x * dist)
                y = int(self.position.y + ray_dir_y * dist)
                
                if not (0 <= x < width and 0 <= y < height):
                    break
                
                # Check obstacles first (they're closer threats)
                for obstacle in obstacle_group:
                    if obstacle.rect.collidepoint(x, y):
                        min_dist = dist
                        hit_obstacle = True
                        break
                
                if hit_obstacle:
                    break
                
                if border_mask.get_at((x, y)):  # Hit wall
                    min_dist = dist
                    collision_point = Vector2(x, y)
                    break
            
            self.bomb_distances[idx] = min_dist
            self.bomb_hit_obstacle[idx] = hit_obstacle
            self.bomb_collision_points[idx] = collision_point

    def draw_rays(self, surface):
        """Visualize rays - green for walls, yellow for bombs"""
        for collision_point in self.wall_collision_points:
            if collision_point:
                pygame.draw.line(surface, GREEN, 
                               (int(self.position.x), int(self.position.y)),
                               (int(collision_point.x), int(collision_point.y)), 1)
                pygame.draw.circle(surface, GREEN, 
                                 (int(collision_point.x), int(collision_point.y)), 2)
        
        for collision_point in self.bomb_collision_points:
            if collision_point:
                pygame.draw.line(surface, YELLOW, 
                               (int(self.position.x), int(self.position.y)),
                               (int(collision_point.x), int(collision_point.y)), 1)
                pygame.draw.circle(surface, YELLOW, 
                                 (int(collision_point.x), int(collision_point.y)), 2)

    def rotate(self, left=False, right=False):
        """Turn the car"""
        if not self.can_move:
            return
        if left:
            self.angle += self.rotation_velocity
        elif right:
            self.angle -= self.rotation_velocity

        # Update image rotation
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        old_center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = old_center
        if left or right:
            self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        """Move car forward based on velocity and angle"""
        if not self.can_move:
            return
        radians = math.radians(self.angle)
        direction = Vector2(math.sin(radians), math.cos(radians))
        self.position -= direction * self.velocity
        self.rect.center = self.position

    def accelerate(self, forward=True):
        """Speed up or slow down"""
        if not self.can_move:
            return
        if forward:
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
        else:
            self.velocity = max(self.velocity - self.acceleration, -self.max_velocity / 2)
        self.move()

    def reduce_speed(self):
        """Gradually slow down when not accelerating"""
        if not self.can_move:
            return
        if self.velocity > 0:
            self.velocity = max(self.velocity - self.acceleration * 0.3, 0)
        elif self.velocity < 0:
            self.velocity = min(self.velocity + self.acceleration * 0.3, 0)
        self.move()

    def reset(self, x=None, y=None):
        """Reset car to starting position"""
        if x is not None and y is not None:
            self.position = Vector2(x, y)
        self.velocity = 0
        self.angle = 0
        self.failed = False
        self.can_move = True
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)
        
        # Reset sensors
        self.wall_distances.fill(self.ray_length)
        self.bomb_distances.fill(self.ray_length)
        self.bomb_hit_obstacle.fill(False)
        self.wall_collision_points = [None] * len(self.wall_ray_angles)
        self.bomb_collision_points = [None] * len(self.bomb_ray_angles)