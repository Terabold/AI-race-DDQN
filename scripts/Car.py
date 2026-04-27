import math
import pygame
import numpy as np
from pygame.math import Vector2
from scripts.Constants import MAXSPEED, ROTATESPEED, ACCELERATION, GREEN, YELLOW
from scripts.ResourceLoader import resource_manager

class Car(pygame.sprite.Sprite):    
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

        self.max_velocity = MAXSPEED
        self.velocity = 0
        self.rotation_velocity = ROTATESPEED
        self.angle = 0
        self.acceleration = ACCELERATION

        self.failed = False
        self.can_move = True

        self.ray_length = 400
        
        # 15 קרניים מ--90 עד +90 מעלות
        self.wall_ray_angles = np.array([-90, -60, -45, -30, -20, -15, -10, 0, 10, 15, 20, 30, 45, 60, 90], dtype=np.float32)
        self.bomb_ray_angles = np.array([-90, -60, -45, -30, -20, -15, -10, 0, 10, 15, 20, 30, 45, 60, 90], dtype=np.float32)
        
        # אתחול המרחקים למקסימום (400 פיקסלים) כברירת מחדל
        # [400.0, 400.0, 400.0...] (15 values)
        self.wall_distances = np.full(len(self.wall_ray_angles), self.ray_length, dtype=np.float32)
        self.bomb_distances = np.full(len(self.bomb_ray_angles), self.ray_length, dtype=np.float32)
        # מערך בוליאני לסימון אם קרן פצצה פגעה במכשול
        # [False, False, False...] (15 values)
        self.bomb_hit_obstacle = np.zeros(len(self.bomb_ray_angles), dtype=bool)
        
        # הכנת רשימה של 15 מקומות ריקים עבור נקודות הפגיעה לציור
        # [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        self.wall_collision_points = [None] * len(self.wall_ray_angles)
        self.bomb_collision_points = [None] * len(self.bomb_ray_angles)
        
        # וקטורי כיוון במערך דו-ממדי
        # [[0, -1], [1, 0], ...]
        self.wall_directions = np.array([
            [math.sin(math.radians(-angle)), -math.cos(math.radians(-angle))]
            for angle in self.wall_ray_angles
        ], dtype=np.float32)
        
        self.bomb_directions = np.array([
            [math.sin(math.radians(-angle)), -math.cos(math.radians(-angle))]
            for angle in self.bomb_ray_angles
        ], dtype=np.float32)

    def cast_rays(self, border_mask, obstacle_group=None):
        # חישוב זווית הרכב עבור הקרניים
        car_rotation = -self.angle
        step = 15  # קפיצות של 15 פיקסלים בבדיקה (כדי שהמשחק לא ייתקע מרוב חישובים)
        width, height = border_mask.get_size()
        
        # יריית קרניים לזיהוי קירות ומכשולים
        self.cast_wall_rays(border_mask, car_rotation, step, width, height)
        if obstacle_group:
            self.cast_bomb_rays(border_mask, obstacle_group, car_rotation, step, width, height)
    
    def cast_wall_rays(self, border_mask, car_rotation, step, width, height):
        # זיהוי קירות בלבד בעזרת 15 קרניים
        
        # איפוס המרחקים למקסימום לפני הבדיקה החדשה
        self.wall_distances.fill(self.ray_length)
        
        # חישוב מראש של סינוס וקוסינוס לשיפור מהירות
        angle_rad = math.radians(car_rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        for idx, base_dir in enumerate(self.wall_directions):
            # סיבוב וקטור הקרן לפי כיוון הרכב
            # x_new = x * cos(a) - y * sin(a)
            # y_new = x * sin(a) + y * cos(a)
            ray_dir_x = base_dir[0] * cos_a - base_dir[1] * sin_a
            ray_dir_y = base_dir[0] * sin_a + base_dir[1] * cos_a
            
            min_dist = self.ray_length
            collision_point = None
            
            # סריקה לאורך הקרן בקפיצות של 15 פיקסלים (15, 30, 45... עד 400)
            for dist in range(step, self.ray_length + 1, step):
                # חישוב המיקום על הקרן לפי מרחק מהרכב
                # (x, y), dist
                x = int(self.position.x + ray_dir_x * dist)
                y = int(self.position.y + ray_dir_y * dist)
                
                # בדיקה אם הנקודה יצאה מגבולות המסך
                if not (0 <= x < width and 0 <= y < height):
                    break
                
                # בדיקה אם הנקודה פגעה בקיר (לפי מסכת הגבולות)
                if border_mask.get_at((x, y)):
                    min_dist = dist
                    collision_point = Vector2(x, y)
                    break # עצירת הסריקה בנקודת הפגיעה הראשונה
            
            # שמירת המרחק ונקודת הפגיעה
            self.wall_distances[idx] = min_dist
            self.wall_collision_points[idx] = collision_point
    
    def cast_bomb_rays(self, border_mask, obstacle_group, car_rotation, step, width, height):
        # שיגור קרניים פצצה: זיהוי קירות + פצצות (15 קרניים)
        # דומה ל-cast_wall_rays אבל עם בדיקה נוספת לפצצות
        self.bomb_distances.fill(self.ray_length)
        self.bomb_hit_obstacle.fill(False)
        
        # חישוב זווית הרכב ברדיאנים
        angle_rad = math.radians(car_rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        for idx, base_dir in enumerate(self.bomb_directions):
            # סיבוב וקטור הקרן לפי כיוון הרכב
            ray_dir_x = base_dir[0] * cos_a - base_dir[1] * sin_a
            ray_dir_y = base_dir[0] * sin_a + base_dir[1] * cos_a
            
            min_dist = self.ray_length
            collision_point = None
            hit_obstacle = False
            
            # סריקה לאורך הקרן בקפיצות של 15 פיקסלים
            for dist in range(step, self.ray_length + 1, step):
                # חישוב נקודה (x, y) על הקרן
                x = int(self.position.x + ray_dir_x * dist)
                y = int(self.position.y + ray_dir_y * dist)
                
                if not (0 <= x < width and 0 <= y < height):
                    break
                
                # עדיפות 1: בדיקה אם פגענו במכשול (פצצה)
                for obstacle in obstacle_group:
                    if obstacle.rect.collidepoint(x, y):
                        min_dist = dist
                        hit_obstacle = True
                        break
                
                if hit_obstacle:
                    break # עצירה אם פגענו במכשול
                
                # עדיפות 2: בדיקה אם פגענו בקיר
                if border_mask.get_at((x, y)):
                    min_dist = dist
                    collision_point = Vector2(x, y)
                    break
            
            # שמירת התוצאות (מרחק, סוג פגיעה ונקודה)
            self.bomb_distances[idx] = min_dist
            self.bomb_hit_obstacle[idx] = hit_obstacle
            self.bomb_collision_points[idx] = collision_point

    def draw_rays(self, surface):
        # ציור קרני הקיר (בירוק)
        for collision_point in self.wall_collision_points:
            if collision_point:
                pygame.draw.line(surface, GREEN, 
                               (int(self.position.x), int(self.position.y)),
                               (int(collision_point.x), int(collision_point.y)), 1)
                pygame.draw.circle(surface, GREEN, 
                                 (int(collision_point.x), int(collision_point.y)), 2)
        
        # ציור קרני הפצצה (בצהוב)
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
        # המרת זווית הרכב לרדיאנים וחישוב וקטור כיוון
        radians = math.radians(self.angle)
        direction = Vector2(math.sin(radians), math.cos(radians))
        # עדכון המיקום (למעלה שלילי)
        # Y, Pygame
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
        # חיכוך - האטה הדרגתית כשלא לוחצים על הגז
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