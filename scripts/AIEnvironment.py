# AIEnvironment.py - WITH FULL VISUALIZATION
import pygame
import math
from scripts.Constants import *
from scripts.Car import Car
from scripts.Obstacle import Obstacle
from scripts.checkpoint import CheckpointManager

class AIEnvironment:
    def __init__(self, surface):
        self.surface = surface
        self.car = Car(*CAR_START_POS, "Red")
        self.num_obstacles = NUM_OBSTACLES
        self.obstacle_group = pygame.sprite.Group()
        self._generate_obstacles()
        self._setup_track()
        self.checkpoint_manager = CheckpointManager()
        self.font = pygame.font.Font(FONT, int(UI_FONT_SIZE/2))
        # Time
        self.max_time = TARGET_TIME
        self.time_remaining = self.max_time
        
        # Episode state
        self.episode_ended = False
        self.car_finished = False
        self.car_crashed = False
        self.car_timeout = False
        
        # DISTANCE TRACKING - SIMPLE
        self.current_checkpoint_distance = 0.0
        self.prev_checkpoint_distance = 0.0
        
        # FOR VISUALIZATION
        self.last_reward = 0.0
        self.last_reward_breakdown = {}
        self.episode_reward = 0.0

    def _setup_track(self):
        self.track_border = pygame.image.load(TRACK_BORDER).convert_alpha()
        self.track_border_mask = pygame.mask.from_surface(self.track_border)
        self.finish_line = pygame.transform.scale(
            pygame.image.load(FINISHLINE).convert_alpha(),
            FINISHLINE_SIZE
        )
        self.finish_line_position = FINISHLINE_POS
        self.finish_mask = pygame.mask.from_surface(self.finish_line)

    def _generate_obstacles(self):
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
        
        # Reset distance tracking
        self.current_checkpoint_distance = 0.0
        self.prev_checkpoint_distance = 0.0
        
        # Reset visualization
        self.last_reward = 0.0
        self.last_reward_breakdown = {}
        self.episode_reward = 0.0

    def get_state(self):
        self.car.cast_rays(self.track_border_mask, self.obstacle_group)
        
        # Checkpoint distance calculation
        if self.checkpoint_manager.current_idx < self.checkpoint_manager.total_checkpoints:
            cp_center = CHECKPOINT_CENTERS[self.checkpoint_manager.current_idx]
            car_pos = np.array([self.car.position.x, self.car.position.y], dtype=np.float32)
            cp_array = np.array(cp_center, dtype=np.float32)
            self.current_checkpoint_distance = np.linalg.norm(cp_array - car_pos)
        else:
            self.current_checkpoint_distance = 0.0
        
        # Normalize rays
        norm_wall_rays = self.car.wall_distances / self.car.ray_length
        norm_bomb_rays = self.car.bomb_distances / self.car.ray_length
        
        norm_vel = max(0.0, self.car.velocity / self.car.max_velocity)
        angle_rad = np.radians(self.car.angle)
        
        # Combine state
        state = np.concatenate([
            norm_wall_rays,              # 15 values
            norm_bomb_rays,              # 15 values
            [norm_vel],                  # 1 value
            [np.sin(angle_rad)],        # 1 value
            [np.cos(angle_rad)]         # 1 value
        ]).astype(np.float32)
        
        return state

    def step(self, action):
        if self.episode_ended:
            return self.get_state(), {
                'collision': False, 'finished': False, 'hit_obstacle': False,
                'timeout': False, 'checkpoint_crossed': False, 'backward_crossed': False
            }, True

        # Store previous distance BEFORE moving
        self.prev_checkpoint_distance = self.current_checkpoint_distance

        pre_velocity = self.car.velocity
        self._handle_car_movement(action)

        car_pos = (self.car.position.x, self.car.position.y)
        crossed, backward = self.checkpoint_manager.check_crossing(car_pos)

        step_info = {
            'collision': False, 'finished': False, 'hit_obstacle': False,
            'timeout': False, 'checkpoint_crossed': crossed, 'backward_crossed': backward
        }

        step_info['hit_obstacle'] = self._check_obstacle(pre_velocity)
        step_info['finished'] = self._check_finish()
        step_info['collision'] = self._check_collision()

        self.time_remaining = max(0, self.time_remaining - 1/FPS)
        if self.time_remaining <= 0 and not self.car_finished and not self.car_crashed:
            self.car.can_move = False
            self.car_timeout = True
            step_info['timeout'] = True
            self.episode_ended = True

        done = self.episode_ended
        next_state = self.get_state()
        return next_state, step_info, done

    def _handle_car_movement(self, action):
        if action is None: return
        moving = action in [1, 2, 5, 6, 7, 8]
        if action in [3, 5, 7]: self.car.rotate(left=True)
        elif action in [4, 6, 8]: self.car.rotate(right=True)
        if action in [1, 5, 6]: self.car.accelerate(True)
        elif action in [2, 7, 8]: self.car.accelerate(False)
        if not moving: self.car.reduce_speed()

    def _check_obstacle(self, pre_velocity):
        for obstacle in self.obstacle_group.sprites():
            if pygame.sprite.collide_mask(self.car, obstacle):
                self.car.velocity *= OBSTACLE_VELOCITY_REDUCTION
                obstacle.kill()
                return pre_velocity > 1.0
        return False

    def _check_finish(self):
        if self.car_finished or self.car_crashed: return False
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

    def _check_collision(self):
        if self.car_crashed: return False
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

    def _draw_text(self, text: str, pos: tuple, color=UI_COLOR, size=UI_FONT_SIZE):
        shadow = self.font.render(text, True, SHADOW_COLOR)
        main   = self.font.render(text, True, color)
        self.surface.blit(shadow, (pos[0]+1, pos[1]+1))
        self.surface.blit(main,   pos)

    def draw(self):
        self.surface.fill((0, 0, 0))
        self.obstacle_group.draw(self.surface)
        self.checkpoint_manager.draw(self.surface)

        if not self.car_finished and not self.car_crashed:
            self.car.draw_rays(self.surface)
            
            # Visualize checkpoint target
            if self.checkpoint_manager.current_idx < self.checkpoint_manager.total_checkpoints:
                cp_center = CHECKPOINT_CENTERS[self.checkpoint_manager.current_idx]
                car_pos = (self.car.position.x, self.car.position.y)
                
                # Line to checkpoint
                pygame.draw.line(self.surface, (255, 255, 0),
                    (int(car_pos[0]), int(car_pos[1])),
                    (int(cp_center[0]), int(cp_center[1])), 2)
                
                # Circle at checkpoint (shrinks as you approach)
                circle_size = int(10 + (self.current_checkpoint_distance / 800) * 30)
                circle_size = max(5, min(circle_size, 40))
                pygame.draw.circle(self.surface, (255, 215, 0),
                    (int(cp_center[0]), int(cp_center[1])), circle_size, 3)
                
                # Distance text near checkpoint
                font = pygame.font.Font(None, 20)
                dist_text = f"{self.current_checkpoint_distance:.0f}px"
                text_surf = font.render(dist_text, True, (255, 255, 255))
                text_bg = pygame.Surface((text_surf.get_width() + 8, text_surf.get_height() + 4))
                text_bg.set_alpha(180)
                text_bg.fill((0, 0, 0))
                text_x = int(cp_center[0]) - text_surf.get_width() // 2
                text_y = int(cp_center[1]) - 30
                self.surface.blit(text_bg, (text_x - 4, text_y - 2))
                self.surface.blit(text_surf, (text_x, text_y))

        self.surface.blit(self.track_border, (0, 0))
        self.surface.blit(self.car.image, self.car.rect)
        self.surface.blit(self.finish_line, self.finish_line_position)

        # === TOP LEFT UI ===
        x, y = MARGIN_X, MARGIN_Y_TOP
        
        # Time
        time_color = GREEN if self.time_remaining > 10 else (YELLOW if self.time_remaining > 3 else RED)
        self._draw_text(f"Time: {self.time_remaining:.1f}s", (x, y), time_color)
        y += LINE_HEIGHT
        
        # Checkpoints
        total_cp = self.checkpoint_manager.total_checkpoints
        current_cp = self.checkpoint_manager.crossed_count
        if self.car_finished: current_cp = total_cp
        self._draw_text(f"CP: {current_cp}/{total_cp}", (x, y))
        y += LINE_HEIGHT
        
        # Speed
        speed_ratio = self.car.velocity / self.car.max_velocity if self.car.max_velocity > 0 else 0
        speed_color = GREEN if speed_ratio > 0.7 else (YELLOW if speed_ratio > 0.3 else RED)
        self._draw_text(f"Speed: {speed_ratio:.1%}", (x, y), speed_color)
        y += LINE_HEIGHT
        
        # Distance to checkpoint
        if self.current_checkpoint_distance > 0:
            dist_color = GREEN if self.current_checkpoint_distance < 200 else (YELLOW if self.current_checkpoint_distance < 400 else WHITE)
            self._draw_text(f"Dist: {self.current_checkpoint_distance:.0f}px", (x, y), dist_color)
            y += LINE_HEIGHT
        
        # Distance delta (progress indicator)
        if self.prev_checkpoint_distance > 0:
            delta = self.prev_checkpoint_distance - self.current_checkpoint_distance
            delta_color = GREEN if delta > 0 else RED
            self._draw_text(f"Δ: {delta:+.1f}px", (x, y), delta_color)
            y += LINE_HEIGHT

        # === REWARD INFO (TOP RIGHT) ===
        rx = self.surface.get_width() - 250
        ry = MARGIN_Y_TOP
        
        # Episode reward
        self._draw_text(f"Episode R: {self.episode_reward:.1f}", (rx, ry), GOLD)
        ry += LINE_HEIGHT
        
        # Last step reward
        reward_color = GREEN if self.last_reward > 0 else (RED if self.last_reward < 0 else WHITE)
        self._draw_text(f"Step R: {self.last_reward:+.2f}", (rx, ry), reward_color)
        ry += LINE_HEIGHT
        
        # Reward breakdown (if exists)
        if self.last_reward_breakdown:
            ry += 5  # Small gap
            for key, value in self.last_reward_breakdown.items():
                if key == "total": continue
                if value == 0: continue  # Skip zero values
                
                color = GREEN if value > 0 else RED
                self._draw_text(f"{key}: {value:+.1f}", (rx, ry), color, size=UI_DEBUG_SIZE)
                ry += DEBUG_LINE_HEIGHT

        # === STATE INFO (BOTTOM LEFT) ===
        state = self.get_state()
        y_debug = self.surface.get_height() - MARGIN_Y_BOTTOM - 10

        # Create background for readability
        info_height = len(state) * DEBUG_LINE_HEIGHT + 20
        info_bg = pygame.Surface((250, info_height))
        info_bg.set_alpha(180)
        info_bg.fill((0, 0, 0))
        self.surface.blit(info_bg, (MARGIN_X - 5, y_debug - info_height))

        y_debug -= info_height - 10

        # State title
        self._draw_text("STATE VECTOR:", (MARGIN_X, y_debug), WHITE, UI_DEBUG_SIZE)
        y_debug += DEBUG_LINE_HEIGHT

        # Display all state values
        for i, value in enumerate(state):
            # Color code by value
            if value < 0.3:
                color = RED
            elif value < 0.6:
                color = YELLOW
            else:
                color = GREEN
            
            self._draw_text(f"[{i:2d}] {value:.3f}", (MARGIN_X, y_debug), color, UI_DEBUG_SIZE)
            y_debug += DEBUG_LINE_HEIGHT