import pygame
import numpy as np
from scripts.Constants import (CAR_START_POS, NUM_OBSTACLES, FINISHLINE_POS, 
                               TARGET_TIME, FPS, OBSTACLE_VELOCITY_REDUCTION,
                               CHECKPOINT_CENTERS, UI_FONT_SIZE, FONT,
                               UI_COLOR, SHADOW_COLOR, MARGIN_X, MARGIN_Y_TOP, LINE_HEIGHT,
                               GREEN, YELLOW, RED, WHITE)
from scripts.Car import Car
from scripts.Obstacle import Obstacle
from scripts.checkpoint import CheckpointManager
from scripts.ResourceLoader import resource_manager


class AIEnvironment:    
    def __init__(self, surface):
        self.surface = surface
        self.car = Car(*CAR_START_POS, "Red")
        self.num_obstacles = NUM_OBSTACLES
        self.obstacle_group = pygame.sprite.Group() # קבוצת ספרייטים לניהול כל המכשולים על המסלול
        self.generate_obstacles()
        self.setup_track()
        self.checkpoint_manager = CheckpointManager()
        self.font = pygame.font.Font(FONT, int(UI_FONT_SIZE / 2))
        
        self.max_time = TARGET_TIME
        self.time_remaining = self.max_time
        
        self.episode_ended = False
        self.car_finished = False
        self.car_crashed = False
        self.car_timeout = False
        
        # חישוב תגמול - שומר מרחקים לצ'קפוינט הנוכחי
        self.current_checkpoint_distance = 0.0
        self.prev_checkpoint_distance = 0.0
        
    def setup_track(self):
        self.track_border = resource_manager.images['track_border']
        self.track_border_mask = resource_manager.track_border_mask
        
        self.finish_line = resource_manager.images['finishline']
        self.finish_line_position = FINISHLINE_POS
        self.finish_mask = resource_manager.images['finishline_mask']

    def generate_obstacles(self):
        obstacle_generator = Obstacle(0, 0, show_image=False)
        self.obstacle_group.add(
            obstacle_generator.generate_obstacles(self.num_obstacles)
        )

    def reset(self):
        # אתחול מצב הסביבה לתחילת סבב חדש
        self.car.reset(*CAR_START_POS)
        obstacle_generator = Obstacle(0, 0, show_image=False)
        obstacle_generator.reshuffle_obstacles(self.obstacle_group, self.num_obstacles)
        self.checkpoint_manager.reset()
        self.time_remaining = self.max_time
        self.episode_ended = False
        self.car_finished = False
        self.car_crashed = False
        self.car_timeout = False
        self.current_checkpoint_distance = 0.0
        self.prev_checkpoint_distance = 0.0

    def get_state(self):
        self.car.cast_rays(self.track_border_mask, self.obstacle_group)
        
        # חישוב המרחק לצ'קפוינט הבא
        if self.checkpoint_manager.current_idx < self.checkpoint_manager.total_checkpoints: # בדיקה אם נשארו עוד צ'קפוינטים לעבור
            cp_center = CHECKPOINT_CENTERS[self.checkpoint_manager.current_idx] 
            # מיקום הרכב כווקטור
            # Vector2(10, 20) -> [10.0, 20.0]
            car_pos = np.array([self.car.position.x, self.car.position.y], dtype=np.float32) 
            # מיקום הצ'קפוינט כווקטור
            # (30, 40) -> [30.0, 40.0]
            cp_array = np.array(cp_center, dtype=np.float32) 
            # חישוב מרחק אווירי (אורך הוקטור שביניהם)
            # norm([20, 20]) -> 28.28
            self.current_checkpoint_distance = np.linalg.norm(cp_array - car_pos) 
        else:
            self.current_checkpoint_distance = 0.0 # אם עבר את כל הצ'קפוינטים
        
        # נרמול טווחי הרדאר לטווח 0-1 (חילוק מרחק ב-400)
        # [0.5, 1.0, 0.2, 0.1, 0.0, 1.0, 0.4, 0.3, 0.8, 0.9, 0.2, 0.5, 0.1, 1.0, 0.7]
        norm_wall_rays = self.car.wall_distances / self.car.ray_length  
        norm_bomb_rays = self.car.bomb_distances / self.car.ray_length  
        norm_vel = max(0.0, self.car.velocity / self.car.max_velocity)  # 1
        angle_rad = np.radians(self.car.angle)  # זוויות ברדיאנים
        
        # הרכבת ה-state vector הסופי (איחוד כל הרשימות לרשימה אחת ארוכה)
        # [d1...d30, vel, sin, cos] (33 values)
        # [0.5] + [1.0] + [0.8] -> [0.5, 1.0, 0.8...]
        state = np.concatenate([
            norm_wall_rays,      # מרחקי קירות (15)
            norm_bomb_rays,      # מרחקי מכשולים (15)
            [norm_vel],          # מהירות נוכחית (1)
            [np.sin(angle_rad)], # כיוון הרכב בסינוס (1)
            [np.cos(angle_rad)]  # כיוון הרכב בקוסינוס (1)
        ]).astype(np.float32)
        
        return state

    def step(self, action):
        # בדיקה ראשונה אם הסבב נגמר
        if self.episode_ended:
            return self.get_state(), {
                'collision': False, 'finished': False, 'hit_obstacle': False,
                'timeout': False, 'checkpoint_crossed': False, 'backward_crossed': False
            }, True

        # שמירת נתונים מקדימה לצורך חישוב תגמול
        self.prev_checkpoint_distance = self.current_checkpoint_distance
        pre_velocity = self.car.velocity

        # ביצוע הפעולה ע"י הרכב
        self.handle_car_movement(action)

        # בדיקת מעבר צ'קפוינטים
        car_pos = (self.car.position.x, self.car.position.y)
        crossed, backward = self.checkpoint_manager.check_crossing(car_pos)

        # הכנת אינפורמציית צעד זמנית
        step_info = {
            'collision': False, 'finished': False, 'hit_obstacle': False,
            'timeout': False, 'checkpoint_crossed': crossed, 'backward_crossed': backward
        }

        # בדיקת תנאים שמסיימים את הסבב
        step_info['hit_obstacle'] = self.check_obstacle(pre_velocity)
        step_info['finished'] = self.check_finish()
        step_info['collision'] = self.check_collision()

        # עדכון טיימר (הורדת זמן יחסית לקצב הפריימים, למשל פחות 0.016 שניות בכל פעם)
        self.time_remaining = max(0, self.time_remaining - 1/FPS)
        
        # בדיקת סיום חריגה בזמן
        if self.time_remaining <= 0 and not self.car_finished and not self.car_crashed:
            self.car.can_move = False
            self.car_timeout = True
            step_info['timeout'] = True
            self.episode_ended = True

        # בדיקת אם הסבב הסתיים
        done = self.episode_ended
        
        # קבלת המצב החדש
        next_state = self.get_state()
        
        return next_state, step_info, done

    def handle_car_movement(self, action):
        if action is None:
            return
        moving = action in [1, 2, 5, 6, 7, 8]
        if action in [3, 5, 7]:
            self.car.rotate(left=True)
        elif action in [4, 6, 8]:
            self.car.rotate(right=True)
        if action in [1, 5, 6]:
            self.car.accelerate(True)
        elif action in [2, 7, 8]:
            self.car.accelerate(False)
        if not moving:
            self.car.reduce_speed()

    def check_obstacle(self, pre_velocity):
        for obstacle in self.obstacle_group.sprites():
            if pygame.sprite.collide_mask(self.car, obstacle):
                self.car.velocity *= OBSTACLE_VELOCITY_REDUCTION
                obstacle.kill()
                return pre_velocity > 1.0
        return False

    def check_finish(self):
        if self.car_finished or self.car_crashed:
            return False
        # חישוב המרחק היחסי (offset) בין הפינה של הרכב לפינה של קו הסיום
        offset = (
            int(self.car.rect.left - self.finish_line_position[0]),
            int(self.car.rect.top - self.finish_line_position[1])
        )
        # בדיקה אם יש חפיפה בין הפיקסלים של הרכב למסכת הסיום במיקום המחושב
        overlap = self.finish_mask.overlap(self.car.mask, offset)
        if overlap:
            if overlap[1] > 2:
                self.car_finished = True
                self.episode_ended = True
                return True
        return False

    def check_collision(self):
        if self.car_crashed:
            return False

        offset = (int(self.car.rect.left), int(self.car.rect.top))

        finish_offset = (
            int(self.car.rect.left - self.finish_line_position[0]),
            int(self.car.rect.top - self.finish_line_position[1])
        )
        
        # בדיקה אם הרכב נוגע בקירות המסלול
        if self.track_border_mask.overlap(self.car.mask, offset):
            self.car.failed = True
            self.car.can_move = False
            self.car_crashed = True
            self.episode_ended = True
            return True
            
        # בדיקה אם הרכב נוגע בקו הסיום מהצד הלא נכון
        overlap = self.finish_mask.overlap(self.car.mask, finish_offset)
        if overlap:
            # נגיעה בחלק העליון של קו הסיום נחשבת התנגשות
            if overlap[1] <= 2: 
                self.car.failed = True
                self.car.can_move = False
                self.car_crashed = True
                self.episode_ended = True
                return True
        return False

    def draw_text(self, text, pos, color=UI_COLOR):
        shadow = self.font.render(text, True, SHADOW_COLOR)
        main = self.font.render(text, True, color)
        self.surface.blit(shadow, (pos[0] + 1, pos[1] + 1))
        self.surface.blit(main, pos)

    def draw(self):
        # ציור כללי של הסביבה
        self.surface.fill((0, 0, 0))
        self.obstacle_group.draw(self.surface)
        self.checkpoint_manager.draw(self.surface)

        if not self.car_finished and not self.car_crashed:
            self.car.draw_rays(self.surface)
            
            # הצגת הצ'קפוינט הבא והמרחק אליו
            if self.checkpoint_manager.current_idx < self.checkpoint_manager.total_checkpoints:
                cp_center = CHECKPOINT_CENTERS[self.checkpoint_manager.current_idx]
                car_pos = (int(self.car.position.x), int(self.car.position.y))
                
                pygame.draw.line(self.surface, (255, 255, 0), car_pos,
                    (int(cp_center[0]), int(cp_center[1])), 2)
                
                # חישוב גודל עיגול משתנה לפי המרחק
                circle_size = int(10 + (self.current_checkpoint_distance / 800) * 30)
                circle_size = max(5, min(circle_size, 40))
                pygame.draw.circle(self.surface, (255, 215, 0),
                    (int(cp_center[0]), int(cp_center[1])), circle_size, 3)

        self.surface.blit(self.track_border, (0, 0))
        self.surface.blit(self.car.image, self.car.rect)
        self.surface.blit(self.finish_line, self.finish_line_position)

        # ממשק משתמש
        x, y = MARGIN_X, MARGIN_Y_TOP
        
        # צבע משתנה לפי הזמן שנותר
        time_color = GREEN if self.time_remaining > 10 else (YELLOW if self.time_remaining > 3 else RED)
        self.draw_text(f"Time: {self.time_remaining:.1f}s", (x, y), time_color)
        y += LINE_HEIGHT
        
        total_cp = self.checkpoint_manager.total_checkpoints
        current_cp = self.checkpoint_manager.crossed_count
        if self.car_finished:
            current_cp = total_cp
        self.draw_text(f"CP: {current_cp}/{total_cp}", (x, y))
        y += LINE_HEIGHT
        
        # חישוב אחוז המהירות הנוכחית
        speed_ratio = self.car.velocity / self.car.max_velocity if self.car.max_velocity > 0 else 0
        speed_color = GREEN if speed_ratio > 0.7 else (YELLOW if speed_ratio > 0.3 else RED)
        self.draw_text(f"Speed: {speed_ratio:.0%}", (x, y), speed_color)
        y += LINE_HEIGHT
        
        if self.current_checkpoint_distance > 0:
            # צבע לפי המרחק מהיעד
            dist_color = GREEN if self.current_checkpoint_distance < 200 else (YELLOW if self.current_checkpoint_distance < 400 else WHITE)
            self.draw_text(f"Dist: {self.current_checkpoint_distance:.0f}px", (x, y), dist_color)