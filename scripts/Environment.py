# סביבת המשחק הראשית - עם סאונד, ספירה לאחור, תמיכה ב-2 שחקנים
import pygame
import numpy as np
from scripts.Constants import (FINISHLINE_POS, CAR_START_POS, CAR1_FAIR_START, CAR2_FAIR_START, 
                               NUM_OBSTACLES, OBSTACLE_VELOCITY_REDUCTION, 
                               TARGET_TIME, FPS, DEFAULT_SOUND_VOLUME)
from scripts.Car import Car
from scripts.Obstacle import Obstacle
from scripts.utils import draw_finished, draw_failed, draw_ui, draw_countdown
from scripts.GameManager import game_state_manager
from scripts.menu import PauseMenu
from scripts.ResourceLoader import resource_manager


class Environment:
    def __init__(self, surface, car_color1=None, car_color2=None):
        self.surface = surface
        self.grass = resource_manager.images['grass']
        self.car1 = None
        self.car2 = None

        self.game_state = "countdown"
        self.previous_state = None

        self.car1_active = car_color1 is not None
        self.car2_active = car_color2 is not None
        self.car1_finished = False
        self.car2_finished = False

        start_x, start_y = CAR_START_POS
        self.setup_cars(start_x, start_y, car_color1, car_color2)

        self.num_obstacles = NUM_OBSTACLES
        self.obstacle_group = pygame.sprite.Group()
        self.generate_obstacles()

        self.setup_track()

        self.car1_time = TARGET_TIME if self.car1_active else 0
        self.car2_time = TARGET_TIME if self.car2_active else 0
        self.remaining_time = max(self.car1_time, self.car2_time)

        self.setup_sound()
        
        self.pause_menu = PauseMenu(surface)

    def setup_cars(self, start_x, start_y, car_color1, car_color2):
        self.all_sprites = pygame.sprite.Group()

        if self.car1_active and self.car2_active:
            # 2 שחקנים - מיקומי פתיחה הוגנים
            self.car1 = Car(*CAR1_FAIR_START, car_color1)
            self.car2 = Car(*CAR2_FAIR_START, car_color2)
        else:
            # שחקן יחיד
            if self.car1_active:
                self.car1 = Car(start_x, start_y, car_color1)
            if self.car2_active:
                self.car2 = Car(start_x, start_y, car_color2)

        if self.car1_active:
            self.all_sprites.add(self.car1)
        if self.car2_active:
            self.all_sprites.add(self.car2)

    def setup_track(self):
        self.track = resource_manager.images['track']
        self.track_border = resource_manager.images['track_border']
        self.track_border_mask = resource_manager.track_border_mask

        self.finish_line = resource_manager.images['finishline']
        self.finish_line_position = FINISHLINE_POS
        self.finish_mask = resource_manager.images['finishline_mask']

    def generate_obstacles(self):
        obstacle_generator = Obstacle(0, 0, show_image=True)
        self.obstacle_group.add(
            obstacle_generator.generate_obstacles(self.num_obstacles)
        )

    def run_countdown(self):
        if self.game_state == "countdown":
            self.countdown_sound.play()
            for i in range(3, 0, -1):
                self.surface.fill((0, 0, 0))
                self.surface.blits((
                    (self.grass, (0, 0)),
                    (self.track, (0, 0)),
                    (self.finish_line, self.finish_line_position),
                ))
                self.obstacle_group.draw(self.surface)
                self.surface.blit(self.track_border, (0, 0))
                self.all_sprites.draw(self.surface)
                draw_countdown(self, i)
                pygame.display.update()
                pygame.time.wait(1000)

            self.game_state = "running"
            self.handle_music(play=True)

    def restart_game(self):
        if self.car1_active:
            if self.car2_active:
                self.car1.reset(*CAR1_FAIR_START)
            else:
                self.car1.reset(*CAR_START_POS)
            self.car1_finished = False
            self.car1_time = TARGET_TIME

        if self.car2_active:
            if self.car1_active:
                self.car2.reset(*CAR2_FAIR_START)
            else:
                self.car2.reset(*CAR_START_POS)
            self.car2_finished = False
            self.car2_time = TARGET_TIME

        self.remaining_time = max(self.car1_time, self.car2_time)

        obstacle_generator = Obstacle(0, 0, show_image=True)
        obstacle_generator.reshuffle_obstacles(self.obstacle_group, self.num_obstacles)

        self.game_state = "countdown"
        self.countdown_sound.stop()
        self.collide_sound.stop()
        self.win_sound.stop()
        self.obstacle_sound.stop()
        self.handle_music(play=False)
        self.run_countdown()

    def check_game_end_condition(self):
        # בדיקה אם שחקן 1 עדיין בתחרות (פעיל, לא סיים, לא התנגש ועדיין יש לו זמן בטיימר)
        car1_racing = self.car1_active and not self.car1_finished and not self.car1.failed and self.car1_time > 0
        # בדיקה דומה לשחקן 2
        car2_racing = self.car2_active and not self.car2_finished and not self.car2.failed and self.car2_time > 0

        # אם אף אחד לא משחק כבר
        if not car1_racing and not car2_racing:
            # אם אחד מהשחקנים סיים
            any_finished = (self.car1_active and self.car1_finished) or (self.car2_active and self.car2_finished) 

            if any_finished:
                self.game_state = "finished"
                self.handle_music(play=False)
            else:
                self.game_state = "failed"
                self.handle_music(play=False)

            return True
        return False

    def update(self):
        if self.game_state == "countdown":
            self.run_countdown()

        elif self.game_state == "running":
            if self.car1_active and not self.car1_finished and not self.car1.failed:
                self.car1_time = max(0, self.car1_time - 1/FPS)
                if self.car1_time <= 0:
                    self.car1.can_move = False

            if self.car2_active and not self.car2_finished and not self.car2.failed:
                self.car2_time = max(0, self.car2_time - 1/FPS)
                if self.car2_time <= 0:
                    self.car2.can_move = False

            # בחירת הזמן הגבוה ביותר מבין שני השחקנים להצגה בטיימר הראשי
            self.remaining_time = max(self.car1_time, self.car2_time)
            self.check_game_end_condition()

    def move(self, action1, action2):
        if self.game_state != "running":
            return False, {}, {}

        car1_info = {'collision': False, 'finished': False, 'hit_obstacle': False}
        car2_info = {'collision': False, 'finished': False, 'hit_obstacle': False}

        # עיבוד פריים בודד של תנועה עבור שני השחקנים
        # מחזיר: (האם נגמר, מידע שחקן 1, מידע שחקן 2)
        # המידע כולל: התנגשות, סיום, או פגיעה במכשול
        if self.car1_active and not self.car1_finished and not self.car1.failed and self.car1_time > 0:
            pre_failed = self.car1.failed
            pre_finished = self.car1_finished
            pre_velocity = self.car1.velocity
            
            self.handle_car_movement(self.car1, action1)
            
            hit_obstacle = self.check_single_car_obstacle(self.car1, pre_velocity)
            just_finished = self.check_single_car_finish(self.car1, pre_finished)
            just_collided = self.check_single_car_collision(self.car1, pre_failed)
            
            car1_info = {
                'collision': just_collided,
                'finished': just_finished,
                'hit_obstacle': hit_obstacle
            }

        # רכב 2
        if self.car2_active and not self.car2_finished and not self.car2.failed and self.car2_time > 0:
            pre_failed = self.car2.failed
            pre_finished = self.car2_finished
            pre_velocity = self.car2.velocity
            
            self.handle_car_movement(self.car2, action2)
            
            hit_obstacle = self.check_single_car_obstacle(self.car2, pre_velocity)
            just_finished = self.check_single_car_finish(self.car2, pre_finished)
            just_collided = self.check_single_car_collision(self.car2, pre_failed)
            
            car2_info = {
                'collision': just_collided,
                'finished': just_finished,
                'hit_obstacle': hit_obstacle
            }

        done = self.check_game_end_condition()
        return done, car1_info, car2_info

    def check_single_car_obstacle(self, car, pre_velocity):
        hit = pygame.sprite.spritecollide(car, self.obstacle_group, True, pygame.sprite.collide_mask)
        if hit:
            car.velocity *= OBSTACLE_VELOCITY_REDUCTION
            self.obstacle_sound.play()
            return pre_velocity > 1.0 # למנוע בעיות במהירות נמוכה במילא האטה שלא נראת לעין
        return False
    
    def toggle_pause(self):
        if self.game_state == "running":
            self.previous_state = self.game_state
            self.game_state = "paused"
            self.handle_music(play=False)
        elif self.game_state == "paused":
            self.game_state = self.previous_state
            self.handle_music(play=True)

    def draw(self):
        # ציור קבוצתי של הרקע, המסלול וקו הסיום (יעיל יותר מציור של כל אחד בנפרד)
        self.surface.blits((
            (self.grass, (0, 0)),
            (self.track, (0, 0)),
            (self.finish_line, self.finish_line_position),
        ))

        self.obstacle_group.draw(self.surface)
        self.surface.blit(self.track_border, (0, 0))
        self.all_sprites.draw(self.surface)

        if self.game_state == "running":
            draw_ui(self)
        elif self.game_state == "finished":
            draw_finished(self)
        elif self.game_state == "failed":
            draw_failed(self)
        elif self.game_state == "paused":
            self.pause_menu.draw()
            
    def check_single_car_finish(self, car, was_finished):
        if was_finished or car.failed: # אם כבר סיים או נפסל אין טעם לבדוק
            return False
        
        # חישוב המרחק היחסי (offset) בין הפינה של הרכב לפינה של קו הסיום
        # offset = (car_x - finish_x, car_y - finish_y)
        car_offset = (int(car.rect.left - self.finish_line_position[0]),
                     int(car.rect.top - self.finish_line_position[1]))
        
        # בדיקת חפיפה בין הפיקסלים של הרכב למסכת הסיום
        overlap = self.finish_mask.overlap(car.mask, car_offset)
        if overlap:
            if overlap[1] > 2: # כלומר במידה ולא מדויק והרכב נכנס פיקסל אחד פנימה בדיקה עדיין תזהה חצייה שגוייה
                if car == self.car1:
                    self.car1_finished = True
                else:
                    self.car2_finished = True
                self.win_sound.play()
                return True
        
        return False

    def check_single_car_collision(self, car, was_failed):
        if was_failed:
            return False
        
        offset = (int(car.rect.left), int(car.rect.top))
        # המרחק היחסי לקו הסיום
        finish_offset = (int(car.rect.left - self.finish_line_position[0]),
                        int(car.rect.top - self.finish_line_position[1]))
        
        if self.track_border_mask.overlap(car.mask, offset):
            car.failed = True
            car.can_move = False
            self.collide_sound.play()
            self.check_game_end_condition()
            return True
        
        overlap = self.finish_mask.overlap(car.mask, finish_offset)
        if overlap:
            if overlap[1] <= 2:
                car.failed = True
                car.can_move = False
                self.collide_sound.play()
                self.check_game_end_condition()
                return True
        
        return False

    def handle_car_movement(self, car, action):
        # החלת הפעולה שנבחרה על הרכב
        # 0=כלום, 1=קדימה, 2=אחורה, 3=שמאלה, 4=ימינה...
        if action is None:
            return

        # פעולות הכוללות תנועה (לא רק סיבוב)
        moving = action in [1, 2, 5, 6, 7, 8]

        if action in [3, 5, 7]:
            car.rotate(left=True)
        elif action in [4, 6, 8]:
            car.rotate(right=True)

        if action in [1, 5, 6]:
            car.accelerate(True)
        elif action in [2, 7, 8]:
            car.accelerate(False)

        if not moving:
            car.reduce_speed()

    def setup_sound(self):
        self.collide_sound = resource_manager.sounds['collision']
        self.win_sound = resource_manager.sounds['win']
        self.obstacle_sound = resource_manager.sounds['obstacle']
        self.countdown_sound = resource_manager.sounds['countdown']

        self.is_music_playing = False
        
        pygame.mixer.music.load(resource_manager.sounds['background_music_path'])
        pygame.mixer.music.set_volume(DEFAULT_SOUND_VOLUME)

    def handle_music(self, play=True):
        if not pygame.mixer.get_init():
            return

        if play:
            # חידוש או התחלת מוזיקה
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
            else:
                pygame.mixer.music.play(-1)
            self.is_music_playing = True
        else:
            # השהיית מוזיקה
            pygame.mixer.music.pause()
            self.is_music_playing = False

    def get_state(self, car_num=1):
        # חישוב מצב לשחקן AI
        if car_num == 1:
            if not self.car1_active:
                return None
            car = self.car1
        elif car_num == 2:
            if not self.car2_active:
                return None
            car = self.car2
        else:
            return None
        
        car.cast_rays(self.track_border_mask, self.obstacle_group)

        norm_wall_rays = car.wall_distances / car.ray_length
        norm_bomb_rays = car.bomb_distances / car.ray_length
        norm_vel = max(0.0, car.velocity / car.max_velocity)
        angle_rad = np.radians(car.angle)
        
        # איחוד כל נתוני החיישנים לרשימה אחת ארוכה עבור הסוכן
        # [d1...d30, speed, sin, cos] (33 values)
        state = np.concatenate([
            norm_wall_rays,
            norm_bomb_rays,
            [norm_vel],
            [np.sin(angle_rad)],
            [np.cos(angle_rad)]
        ]).astype(np.float32)
        
        return state