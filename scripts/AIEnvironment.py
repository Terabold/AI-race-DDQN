# training environment - simplified version without sounds/countdown
# runs thousands of episodes fast
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
    """
    Simplified training environment - no sounds, no countdown.
    
    Runs thousands of episodes fast for training the DQN agent.
    Single car only, uses checkpoints for reward calculation.
    """
    
    def __init__(self, surface):
        self.surface = surface
        self.car = Car(*CAR_START_POS, "Red")
        self.num_obstacles = NUM_OBSTACLES
        self.obstacle_group = pygame.sprite.Group()
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
        
        # for reward calc
        self.current_checkpoint_distance = 0.0
        self.prev_checkpoint_distance = 0.0
        
    def setup_track(self):
        # Get all track assets from ResourceManager (loaded once at startup)
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
        # 33 numbers: 15 wall rays + 15 bomb rays + vel + sin + cos
        self.car.cast_rays(self.track_border_mask, self.obstacle_group)
        
        # distance to next checkpoint
        if self.checkpoint_manager.current_idx < self.checkpoint_manager.total_checkpoints:
            cp_center = CHECKPOINT_CENTERS[self.checkpoint_manager.current_idx]
            car_pos = np.array([self.car.position.x, self.car.position.y], dtype=np.float32)
            cp_array = np.array(cp_center, dtype=np.float32)
            self.current_checkpoint_distance = np.linalg.norm(cp_array - car_pos)
        else:
            self.current_checkpoint_distance = 0.0
        
        # normalize 0-1
        norm_wall_rays = self.car.wall_distances / self.car.ray_length
        norm_bomb_rays = self.car.bomb_distances / self.car.ray_length
        norm_vel = max(0.0, self.car.velocity / self.car.max_velocity)
        angle_rad = np.radians(self.car.angle)
        
        state = np.concatenate([
            norm_wall_rays,      # 15
            norm_bomb_rays,      # 15
            [norm_vel],          # 1
            [np.sin(angle_rad)], # 1
            [np.cos(angle_rad)]  # 1
        ]).astype(np.float32)
        
        return state

    def step(self, action):
        # one frame - returns (state, info, done)
        if self.episode_ended:
            return self.get_state(), {
                'collision': False, 'finished': False, 'hit_obstacle': False,
                'timeout': False, 'checkpoint_crossed': False, 'backward_crossed': False
            }, True

        # save distance BEFORE moving
        self.prev_checkpoint_distance = self.current_checkpoint_distance

        pre_velocity = self.car.velocity
        self.handle_car_movement(action)

        # check checkpoints
        car_pos = (self.car.position.x, self.car.position.y)
        crossed, backward = self.checkpoint_manager.check_crossing(car_pos)

        step_info = {
            'collision': False, 'finished': False, 'hit_obstacle': False,
            'timeout': False, 'checkpoint_crossed': crossed, 'backward_crossed': backward
        }

        step_info['hit_obstacle'] = self.check_obstacle(pre_velocity)
        step_info['finished'] = self.check_finish()
        step_info['collision'] = self.check_collision()

        # timer
        self.time_remaining = max(0, self.time_remaining - 1/FPS)
        if self.time_remaining <= 0 and not self.car_finished and not self.car_crashed:
            self.car.can_move = False
            self.car_timeout = True
            step_info['timeout'] = True
            self.episode_ended = True

        done = self.episode_ended
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
        offset = (
            int(self.car.rect.left - self.finish_line_position[0]),
            int(self.car.rect.top - self.finish_line_position[1])
        )
        if overlap := self.finish_mask.overlap(self.car.mask, offset):
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
        if self.track_border_mask.overlap(self.car.mask, offset):
            self.car.failed = True
            self.car.can_move = False
            self.car_crashed = True
            self.episode_ended = True
            return True
        if overlap := self.finish_mask.overlap(self.car.mask, finish_offset):
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
        # render for training videos
        self.surface.fill((0, 0, 0))
        self.obstacle_group.draw(self.surface)
        self.checkpoint_manager.draw(self.surface)

        if not self.car_finished and not self.car_crashed:
            self.car.draw_rays(self.surface)
            
            # line to next checkpoint
            if self.checkpoint_manager.current_idx < self.checkpoint_manager.total_checkpoints:
                cp_center = CHECKPOINT_CENTERS[self.checkpoint_manager.current_idx]
                car_pos = (int(self.car.position.x), int(self.car.position.y))
                
                pygame.draw.line(self.surface, (255, 255, 0), car_pos,
                    (int(cp_center[0]), int(cp_center[1])), 2)
                
                circle_size = int(10 + (self.current_checkpoint_distance / 800) * 30)
                circle_size = max(5, min(circle_size, 40))
                pygame.draw.circle(self.surface, (255, 215, 0),
                    (int(cp_center[0]), int(cp_center[1])), circle_size, 3)

        self.surface.blit(self.track_border, (0, 0))
        self.surface.blit(self.car.image, self.car.rect)
        self.surface.blit(self.finish_line, self.finish_line_position)

        # ui
        x, y = MARGIN_X, MARGIN_Y_TOP
        
        time_color = GREEN if self.time_remaining > 10 else (YELLOW if self.time_remaining > 3 else RED)
        self.draw_text(f"Time: {self.time_remaining:.1f}s", (x, y), time_color)
        y += LINE_HEIGHT
        
        total_cp = self.checkpoint_manager.total_checkpoints
        current_cp = self.checkpoint_manager.crossed_count
        if self.car_finished:
            current_cp = total_cp
        self.draw_text(f"CP: {current_cp}/{total_cp}", (x, y))
        y += LINE_HEIGHT
        
        speed_ratio = self.car.velocity / self.car.max_velocity if self.car.max_velocity > 0 else 0
        speed_color = GREEN if speed_ratio > 0.7 else (YELLOW if speed_ratio > 0.3 else RED)
        self.draw_text(f"Speed: {speed_ratio:.0%}", (x, y), speed_color)
        y += LINE_HEIGHT
        
        if self.current_checkpoint_distance > 0:
            dist_color = GREEN if self.current_checkpoint_distance < 200 else (YELLOW if self.current_checkpoint_distance < 400 else WHITE)
            self.draw_text(f"Dist: {self.current_checkpoint_distance:.0f}px", (x, y), dist_color)