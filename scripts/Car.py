# הרכב - פיזיקה + חיישני קרינה
import math
import pygame
import numpy as np
from pygame.math import Vector2
from scripts.Constants import MAXSPEED, ROTATESPEED, ACCELERATION, GREEN, YELLOW
from scripts.ResourceLoader import resource_manager

class Car(pygame.sprite.Sprite):
    """
    The player's car - handles physics, movement, and AI raycasting sensors.
    
    Raycasting: 15 rays (-90 to +90 degrees) detect walls and obstacles.
    Wall rays = detect borders only, bomb rays = detect borders + obstacles.
    Used by AI to "see" the track.
    """
    
    def __init__(self, x, y, car_color="Red"):
        super().__init__()
        self.position = Vector2(x, y)
        self.car_color = car_color

        # טעינת תמונת הרכב מה-ResourceManager
        original_img = resource_manager.images[f'car_{car_color}_original']
        # סקייל לגודל משחק
        self.image = pygame.transform.scale(original_img, (19, 38))
        self.original_image = self.image
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)  # זיהוי התנגשות ברמת פיקסל

        # פיזיקה
        self.max_velocity = MAXSPEED
        self.velocity = 0
        self.rotation_velocity = ROTATESPEED
        self.angle = 0
        self.acceleration = ACCELERATION

        self.failed = False
        self.can_move = True

        # מערכת קרינה - כמו סונאר, יורים קרניים לכל הכיוונים
        # wall rays = זיהוי קירות בלבד
        # bomb rays = זיהוי קירות + פצצות
        self.ray_length = 400
        
        # 15 קרניים מ--90 עד +90 מעלות
        self.wall_ray_angles = np.array([-90, -60, -45, -30, -20, -15, -10, 0, 10, 15, 20, 30, 45, 60, 90], dtype=np.float32)
        self.bomb_ray_angles = np.array([-90, -60, -45, -30, -20, -15, -10, 0, 10, 15, 20, 30, 45, 60, 90], dtype=np.float32)
        
        # קריאות מרחקים
        self.wall_distances = np.full(len(self.wall_ray_angles), self.ray_length, dtype=np.float32)
        self.bomb_distances = np.full(len(self.bomb_ray_angles), self.ray_length, dtype=np.float32)
        self.bomb_hit_obstacle = np.zeros(len(self.bomb_ray_angles), dtype=bool)
        
        # לציור הקרניים
        self.wall_collision_points = [None] * len(self.wall_ray_angles)
        self.bomb_collision_points = [None] * len(self.bomb_ray_angles)
        
        # חישוב מראש של כיוונים למהירות
        self.wall_directions = np.array([
            [math.sin(math.radians(-angle)), -math.cos(math.radians(-angle))]
            for angle in self.wall_ray_angles
        ], dtype=np.float32)
        
        self.bomb_directions = np.array([
            [math.sin(math.radians(-angle)), -math.cos(math.radians(-angle))]
            for angle in self.bomb_ray_angles
        ], dtype=np.float32)

    def cast_rays(self, border_mask, obstacle_group=None):
        """
        Shoot sensor rays to detect surroundings.
        Results stored in wall_distances and bomb_distances arrays (0-400 pixels).
        """
        car_rotation = -self.angle
        step = 15  # בדיקה כל 15 פיקסלים לביצועים טובים
        width, height = border_mask.get_size()
        
        self.cast_wall_rays(border_mask, car_rotation, step, width, height)
        if obstacle_group:
            self.cast_bomb_rays(border_mask, obstacle_group, car_rotation, step, width, height)
    
    def cast_wall_rays(self, border_mask, car_rotation, step, width, height):
        # קירות בלבד
        self.wall_distances.fill(self.ray_length)
        
        angle_rad = math.radians(car_rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        for idx, base_dir in enumerate(self.wall_directions):
            # סיבוב לפי זווית הרכב
            ray_dir_x = base_dir[0] * cos_a - base_dir[1] * sin_a
            ray_dir_y = base_dir[0] * sin_a + base_dir[1] * cos_a
            
            min_dist = self.ray_length
            collision_point = None
            
            # מעבר לאורך הקרן
            for dist in range(step, self.ray_length + 1, step):
                x = int(self.position.x + ray_dir_x * dist)
                y = int(self.position.y + ray_dir_y * dist)
                
                if not (0 <= x < width and 0 <= y < height):
                    break
                
                if border_mask.get_at((x, y)):  # פגע בקיר
                    min_dist = dist
                    collision_point = Vector2(x, y)
                    break
            
            self.wall_distances[idx] = min_dist
            self.wall_collision_points[idx] = collision_point
    
    def cast_bomb_rays(self, border_mask, obstacle_group, car_rotation, step, width, height):
        # קירות + פצצות
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
                
                # בדיקת פצצות קודם
                for obstacle in obstacle_group:
                    if obstacle.rect.collidepoint(x, y):
                        min_dist = dist
                        hit_obstacle = True
                        break
                
                if hit_obstacle:
                    break
                
                if border_mask.get_at((x, y)):
                    min_dist = dist
                    collision_point = Vector2(x, y)
                    break
            
            self.bomb_distances[idx] = min_dist
            self.bomb_hit_obstacle[idx] = hit_obstacle
            self.bomb_collision_points[idx] = collision_point

    def draw_rays(self, surface):
        # ירוק = קרני קיר, צהוב = קרני פצצה
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
        if not self.can_move:
            return
        if left:
            self.angle += self.rotation_velocity
        elif right:
            self.angle -= self.rotation_velocity

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        old_center = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = old_center
        if left or right:
            self.mask = pygame.mask.from_surface(self.image)

    def move(self):
        if not self.can_move:
            return
        radians = math.radians(self.angle)
        direction = Vector2(math.sin(radians), math.cos(radians))
        self.position -= direction * self.velocity
        self.rect.center = self.position

    def accelerate(self, forward=True):
        if not self.can_move:
            return
        if forward:
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
        else:
            self.velocity = max(self.velocity - self.acceleration, -self.max_velocity / 2)
        self.move()

    def reduce_speed(self):
        # חיכוך כשלא לוחצים גז
        if not self.can_move:
            return
        if self.velocity > 0:
            self.velocity = max(self.velocity - self.acceleration * 0.3, 0)
        elif self.velocity < 0:
            self.velocity = min(self.velocity + self.acceleration * 0.3, 0)
        self.move()

    def reset(self, x=None, y=None):
        if x is not None and y is not None:
            self.position = Vector2(x, y)
        self.velocity = 0
        self.angle = 0
        self.failed = False
        self.can_move = True
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)
        
        # איפוס חיישנים
        self.wall_distances.fill(self.ray_length)
        self.bomb_distances.fill(self.ray_length)
        self.bomb_hit_obstacle.fill(False)
        self.wall_collision_points = [None] * len(self.wall_ray_angles)
        self.bomb_collision_points = [None] * len(self.bomb_ray_angles)