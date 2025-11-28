import pygame
import torch
import os
import numpy as np
from scripts.AIEnvironment import AIEnvironment
from scripts.dqn_agent import DQNAgent
from scripts.Constants import FONT, INFERENCE_EPSILON, BOMB_LIST, NUM_OBSTACLES, GOLD, YELLOW, GREEN, RED, ORANGE, WHITE
from scripts.Obstacle import Obstacle
from scripts.GameManager import game_state_manager

class TesterGame:
    """Run AI players with unique bomb presets - fully replayable"""
    
    def __init__(self, display, clock):
        self.display = display
        self.clock = clock
        self.environments = []
        self.active_indices = set()
        self.agent = None
        self.initialized = False
        self.test_running = False
        
        # Configuration
        self.total_cars = 100
        self.current_seed = 0  # Changes each run
        
        # Fate tracking
        self.fates = {'finished': 0, 'crashed': 0, 'timeout': 0}
        self.font_small = pygame.font.Font(FONT, 18)
        self.font_med = pygame.font.Font(FONT, 22)
        
    def initialize(self):
        """Load agent once (reuse for all tests)"""
        if self.initialized:
            return
        
        print(f"\nLoading AI model...")
        pygame.mixer.quit()
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load single agent once
        self.agent = DQNAgent(device=device)
        if os.path.exists(self.agent.model_path):
            self.agent.load_model(self.agent.model_path)
            print(f"Loaded model\n")
        else:
            print("Warning: No trained model found!")
        
        self.agent.epsilon = INFERENCE_EPSILON
        self.agent.policy_net.eval()
        self.agent.target_net.eval()
        
        self.initialized = True
    
    def start_test(self, num_cars=None):
        """Start a fresh test run with new obstacle seeds"""
        if num_cars is not None:
            self.total_cars = num_cars
        
        print(f"\nStarting test with {self.total_cars} cars...")
        
        # Reset state
        self.environments.clear()
        self.active_indices.clear()
        self.fates = {'finished': 0, 'crashed': 0, 'timeout': 0}
        self.test_running = True
        self.current_seed += 1
        
        # Create fresh environments with new obstacle seeds
        np.random.seed(self.current_seed)
        base_positions = np.array(BOMB_LIST)
        
        for i in range(self.total_cars):
            env = AIEnvironment(self.display)
            
            # Unique shuffled obstacles for each car
            shuffled_indices = np.random.permutation(len(base_positions))[:NUM_OBSTACLES]
            shuffled_positions = base_positions[shuffled_indices]
            
            # Replace all obstacles
            env.obstacle_group.empty()
            for x, y in shuffled_positions:
                obs = Obstacle(int(x), int(y), show_image=False)
                env.obstacle_group.add(obs)
            
            self.environments.append(env)
            self.active_indices.add(i)
        
        # Cache reference environment for drawing
        self.ref_env = self.environments[0]
        
        print(f"Ready: {self.total_cars} players (seed: {self.current_seed})\n")
    
    def run(self, dt):
        if game_state_manager.getState() != 'tester':
            return
        
        # Initialize agent on first run
        if not self.initialized:
            self.initialize()
        
        # Start test if not running
        if not self.test_running:
            self.start_test()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.test_running = False
                    game_state_manager.setState('tester_settings')
                    return
                elif event.key == pygame.K_r:  # Restart current test
                    self.start_test()
                    return
        
        # Step ONLY active environments
        finished_this_frame = []
        for i in self.active_indices:
            env = self.environments[i]
            
            state = env.get_state()
            if state is None:
                continue
            
            action = self.agent.get_action(state, training=False)
            next_state, step_info, done = env.step(action)
            
            if done:
                self.record_fate(env)
                finished_this_frame.append(i)
        
        # Remove finished cars from active set
        for i in finished_this_frame:
            self.active_indices.discard(i)
        
        # Draw
        self.display.fill((0, 0, 0))
        self.draw_track()
        self.draw_cars()
        self.draw_stats()
    
    def record_fate(self, env):
        """Record what happened to a car"""
        if env.car_finished:
            self.fates['finished'] += 1
        elif env.car_crashed:
            self.fates['crashed'] += 1
        elif env.car_timeout:
            self.fates['timeout'] += 1
    
    def draw_track(self):
        """Draw track and finish line once"""
        self.display.blit(self.ref_env.track_border, (0, 0))
        self.display.blit(self.ref_env.finish_line, self.ref_env.finish_line_position)
    
    def draw_cars(self):
        """Draw only ACTIVE car positions"""
        for i in self.active_indices:
            car = self.environments[i].car
            self.display.blit(car.image, car.rect)
    
    def draw_stats(self):
        """Draw fate counter on screen"""
        y = 15
        x = self.display.get_width() - 280
        
        # Title
        title = self.font_med.render("TEST RESULTS", True, GOLD)
        self.display.blit(title, (x, y))
        y += 32
        
        # Active
        active_text = self.font_small.render(f"Active: {len(self.active_indices)}/{self.total_cars}", True, YELLOW)
        self.display.blit(active_text, (x, y))
        y += 26
        
        # Finished
        fin_text = self.font_small.render(f"Finished: {self.fates['finished']}", True, GREEN)
        self.display.blit(fin_text, (x, y))
        y += 26
        
        # Crashed
        crash_text = self.font_small.render(f"Crashed: {self.fates['crashed']}", True, RED)
        self.display.blit(crash_text, (x, y))
        y += 26
        
        # Timeout
        timeout_text = self.font_small.render(f"Timeout: {self.fates['timeout']}", True, ORANGE)
        self.display.blit(timeout_text, (x, y))
        y += 26
        
        # Progress
        completed = sum(self.fates.values())
        pct = int((completed / self.total_cars) * 100) if self.total_cars > 0 else 0
        prog_text = self.font_small.render(f"Progress: {pct}%", True, WHITE)
        self.display.blit(prog_text, (x, y))
        
        # Info text at bottom
        info_y = self.display.get_height() - 60
        info_text = self.font_small.render("R: Restart  |  ESC: Menu", True, (200, 200, 200))
        self.display.blit(info_text, (x, info_y))
        
        # Done
        if len(self.active_indices) == 0 and completed > 0:
            done_text = self.font_med.render("COMPLETE", True, GREEN)
            self.display.blit(done_text, (x, info_y - 40))