# TesterEnvironment.py - Handles multiple AI cars for performance testing
import pygame
import numpy as np
from scripts.Constants import *
from scripts.Car import Car
from scripts.Obstacle import Obstacle

class TestCar:
    """Wrapper for a single test car with its own state"""
    def __init__(self, car_id, color="Red"):
        self.id = car_id
        self.car = Car(*CAR_START_POS, color)
        
        # Create unique obstacle group for this car
        self.obstacle_group = pygame.sprite.Group()
        obstacle_generator = Obstacle(0, 0, show_image=True)
        self.obstacle_group.add(
            obstacle_generator.generate_obstacles(NUM_OBSTACLES)
        )
        
        # Status tracking
        self.active = True
        self.finished = False
        self.crashed = False
        self.timeout = False
        self.time_remaining = TARGET_TIME
        self.checkpoints_crossed = 0
        self.finish_time = 0.0
        
    def reset(self):
        """Reset car to starting position"""
        self.car.reset(*CAR_START_POS)
        self.active = True
        self.finished = False
        self.crashed = False
        self.timeout = False
        self.time_remaining = TARGET_TIME
        self.checkpoints_crossed = 0
        self.finish_time = 0.0
        
        # Regenerate obstacles
        obstacle_generator = Obstacle(0, 0, show_image=True)
        obstacle_generator.reshuffle_obstacles(self.obstacle_group, NUM_OBSTACLES)


class TesterEnvironment:
    """Environment for testing multiple AI cars simultaneously"""
    
    def __init__(self, surface, num_cars=10):
        self.surface = surface
        self.num_cars = num_cars
        
        # Create all test cars with different colors
        colors = list(CAR_COLORS.keys())
        self.test_cars = [
            TestCar(i, colors[i % len(colors)]) 
            for i in range(num_cars)
        ]
        
        # Track setup
        self._setup_track()
        
        # Statistics
        self.total_finished = 0
        self.total_crashed = 0
        self.total_timeout = 0
        self.total_active = num_cars
        self.fastest_time = 0.0
        self.fastest_car_id = -1
        
        # Timing
        self.test_completed = False
        self.frame_count = 0
        
        # Font for stats
        self.font = pygame.font.Font(FONT, 24)
        self.small_font = pygame.font.Font(FONT, 18)
        
    def _setup_track(self):
        """Load track assets"""
        self.track = pygame.image.load(TRACK).convert_alpha()
        self.track_border = pygame.image.load(TRACK_BORDER).convert_alpha()
        self.track_border_mask = pygame.mask.from_surface(self.track_border)
        
        self.finish_line = pygame.transform.scale(
            pygame.image.load(FINISHLINE).convert_alpha(),
            FINISHLINE_SIZE
        )
        self.finish_line_position = FINISHLINE_POS
        self.finish_mask = pygame.mask.from_surface(self.finish_line)
        
    def reset(self):
        """Reset all cars and statistics"""
        for test_car in self.test_cars:
            test_car.reset()
            
        self.total_finished = 0
        self.total_crashed = 0
        self.total_timeout = 0
        self.total_active = self.num_cars
        self.fastest_time = 0.0
        self.fastest_car_id = -1
        self.test_completed = False
        self.frame_count = 0
        
    def get_state(self, car_id):
        """Get state for a specific car"""
        if car_id >= len(self.test_cars):
            return None
            
        test_car = self.test_cars[car_id]
        if not test_car.active:
            return None
            
        car = test_car.car
        
        # Cast rays for this car
        car.cast_rays(self.track_border_mask, test_car.obstacle_group)
        
        # Normalize rays
        norm_wall_rays = car.wall_distances / car.ray_length
        norm_bomb_rays = car.bomb_distances / car.ray_length
        
        # Normalized velocity
        norm_vel = max(0.0, car.velocity / car.max_velocity)
        
        # Car orientation
        angle_rad = np.radians(car.angle)
        
        # Combine state
        state = np.concatenate([
            norm_wall_rays,              # 15 values
            norm_bomb_rays,              # 15 values
            [norm_vel],                  # 1 value
            [np.sin(angle_rad)],        # 1 value
            [np.cos(angle_rad)]         # 1 value
        ]).astype(np.float32)
        
        return state
    
    def step(self, car_id, action):
        """Execute action for a specific car"""
        if car_id >= len(self.test_cars):
            return False
            
        test_car = self.test_cars[car_id]
        if not test_car.active:
            return False
            
        car = test_car.car
        
        # Handle movement
        self._handle_car_movement(car, action)
        
        # Update time
        test_car.time_remaining = max(0, test_car.time_remaining - 1/FPS)
        
        # Check obstacles (car-specific)
        self._check_obstacle(test_car)
        
        # Check finish
        if self._check_finish(test_car):
            test_car.finished = True
            test_car.active = False
            test_car.finish_time = TARGET_TIME - test_car.time_remaining
            self.total_finished += 1
            self.total_active -= 1
            
            # Track fastest
            if self.fastest_time == 0 or test_car.finish_time < self.fastest_time:
                self.fastest_time = test_car.finish_time
                self.fastest_car_id = car_id
            
            return False
        
        # Check collision
        if self._check_collision(test_car):
            test_car.crashed = True
            test_car.active = False
            self.total_crashed += 1
            self.total_active -= 1
            return False
        
        # Check timeout
        if test_car.time_remaining <= 0 and not test_car.finished:
            car.can_move = False
            test_car.timeout = True
            test_car.active = False
            self.total_timeout += 1
            self.total_active -= 1
            return False
        
        return True  # Car still active
    
    def _handle_car_movement(self, car, action):
        """Execute movement action"""
        if action is None:
            return
            
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
    
    def _check_obstacle(self, test_car):
        """Check obstacle collision for specific car"""
        car = test_car.car
        for obstacle in test_car.obstacle_group.sprites():
            if pygame.sprite.collide_mask(car, obstacle):
                car.velocity *= OBSTACLE_VELOCITY_REDUCTION
                obstacle.kill()
                return True
        return False
    
    def _check_finish(self, test_car):
        """Check if car crossed finish line"""
        if test_car.finished or test_car.crashed:
            return False
            
        car = test_car.car
        offset = (
            int(car.rect.left - self.finish_line_position[0]),
            int(car.rect.top - self.finish_line_position[1])
        )
        
        if overlap := self.finish_mask.overlap(car.mask, offset):
            if overlap[1] > 2:
                return True
        return False
    
    def _check_collision(self, test_car):
        """Check if car crashed into wall"""
        if test_car.crashed:
            return False
            
        car = test_car.car
        offset = (int(car.rect.left), int(car.rect.top))
        
        if self.track_border_mask.overlap(car.mask, offset):
            car.failed = True
            car.can_move = False
            return True
        
        # Check finish line collision (front collision)
        finish_offset = (
            int(car.rect.left - self.finish_line_position[0]),
            int(car.rect.top - self.finish_line_position[1])
        )
        
        if overlap := self.finish_mask.overlap(car.mask, finish_offset):
            if overlap[1] <= 2:
                car.failed = True
                car.can_move = False
                return True
        
        return False
    
    def is_test_complete(self):
        """Check if all cars have finished/failed"""
        return self.total_active == 0
    
    def draw(self):
        """Draw the testing environment"""
        # Background
        self.surface.blit(self.track, (0, 0))
        
        # Draw only active or finished cars (not crashed/timeout)
        for test_car in self.test_cars:
            if test_car.finished or test_car.active:
                self.surface.blit(test_car.car.image, test_car.car.rect)
        
        # Draw track border and finish line on top
        self.surface.blit(self.track_border, (0, 0))
        self.surface.blit(self.finish_line, self.finish_line_position)
        
        # Draw statistics overlay
        self._draw_stats()
    
    def _draw_stats(self):
        """Draw statistics overlay"""
        # Create semi-transparent background
        stats_bg = pygame.Surface((400, 250))
        stats_bg.set_alpha(200)
        stats_bg.fill((0, 0, 0))
        self.surface.blit(stats_bg, (10, 10))
        
        # Title
        title = self.font.render("AI Performance Test", True, (255, 215, 0))
        self.surface.blit(title, (20, 20))
        
        y = 60
        line_height = 30
        
        # Statistics
        stats = [
            f"Total Cars: {self.num_cars}",
            f"Active: {self.total_active}",
            f"Finished: {self.total_finished} ({self._percent(self.total_finished)}%)",
            f"Crashed: {self.total_crashed} ({self._percent(self.total_crashed)}%)",
            f"Timeout: {self.total_timeout} ({self._percent(self.total_timeout)}%)",
        ]
        
        for stat in stats:
            text = self.small_font.render(stat, True, WHITE)
            self.surface.blit(text, (20, y))
            y += line_height
        
        # Fastest time
        if self.fastest_time > 0:
            fastest_text = f"Fastest: Car #{self.fastest_car_id} - {self.fastest_time:.2f}s"
            text = self.small_font.render(fastest_text, True, GREEN)
            self.surface.blit(text, (20, y))
        
        # Completion message
        if self.is_test_complete():
            complete_text = "TEST COMPLETE - Press SPACE"
            text = self.font.render(complete_text, True, (255, 255, 0))
            text_rect = text.get_rect(center=(WIDTH//2, HEIGHT - 50))
            
            # Pulsing effect
            alpha = int(128 + 127 * np.sin(self.frame_count * 0.1))
            text.set_alpha(alpha)
            self.surface.blit(text, text_rect)
        
        self.frame_count += 1
    
    def _percent(self, value):
        """Calculate percentage"""
        if self.num_cars == 0:
            return 0
        return int((value / self.num_cars) * 100)
    
    def get_summary(self):
        """Get test summary statistics"""
        return {
            'total_cars': self.num_cars,
            'finished': self.total_finished,
            'crashed': self.total_crashed,
            'timeout': self.total_timeout,
            'finish_rate': self._percent(self.total_finished),
            'crash_rate': self._percent(self.total_crashed),
            'timeout_rate': self._percent(self.total_timeout),
            'fastest_time': self.fastest_time,
            'fastest_car_id': self.fastest_car_id
        }