# TesterGame.py - AI Performance Testing System
import pygame
import sys
import os
from scripts.TesterEnvironment import TesterEnvironment
from scripts.dqn_agent import DQNAgent
from scripts.GameManager import game_state_manager
from scripts.Constants import *

class TesterGame:
    """Manages AI performance testing with multiple cars"""
    
    def __init__(self, display, clock):
        self.display = display
        self.clock = clock
        self.environment = None
        self.agent = None
        self.initialized = False
        self.num_cars = 10  # Default, will be set by menu
        
        # Testing state
        self.test_running = False
        self.test_complete = False
        
        # Font
        self.font = pygame.font.Font(FONT, 28)
        self.small_font = pygame.font.Font(FONT, 20)
        
    def initialize(self):
        """Initialize the testing environment"""
        if self.initialized:
            return
        
        # Get number of cars from game manager
        self.num_cars = getattr(game_state_manager, 'tester_num_cars', 10)
        
        print("\n" + "="*80)
        print(f"AI PERFORMANCE TEST - {self.num_cars} Cars")
        print("="*80)
        
        # Create environment
        self.environment = TesterEnvironment(self.display, self.num_cars)
        
        # Load AI agent
        self.agent = DQNAgent()
        
        if not os.path.exists(self.agent.model_path):
            print("ERROR: No trained model found!")
            print(f"Expected at: {self.agent.model_path}")
            print("Please train a model first.")
            return False
        
        # Load model
        if not self.agent.load_model(self.agent.model_path):
            print("ERROR: Failed to load model!")
            return False
        
        # Set to pure exploitation (no exploration)
        self.agent.epsilon = 0.0
        self.agent.policy_net.eval()
        self.agent.target_net.eval()
        
        print(f"Model loaded successfully")
        print(f"Testing {self.num_cars} AI cars simultaneously")
        print(f"Epsilon: {self.agent.epsilon} (pure exploitation)")
        print("\nControls:")
        print("  SPACE - Start test / Restart test")
        print("  ESC - Return to menu")
        print("="*80 + "\n")
        
        self.initialized = True
        return True
    
    def start_test(self):
        """Start or restart the test"""
        self.environment.reset()
        self.test_running = True
        self.test_complete = False
        print(f"\nTest started with {self.num_cars} cars")
    
    def run(self, dt):
        """Main test loop"""
        if game_state_manager.getState() != 'tester':
            return
        
        # Initialize if needed
        if not self.initialized:
            if not self.initialize():
                # Failed to initialize, return to menu
                game_state_manager.setState('menu')
                return
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._save_and_exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._return_to_menu()
                elif event.key == pygame.K_SPACE:
                    if not self.test_running or self.test_complete:
                        self.start_test()
        
        # Run test
        if self.test_running and not self.test_complete:
            self._run_test_step()
        
        # Draw
        self.environment.draw()
        
        # Show instructions if not running
        if not self.test_running:
            self._draw_start_message()
        
        pygame.display.update()
        self.clock.tick(FPS)
    
    def _run_test_step(self):
        """Execute one step of the test for all active cars"""
        # Process each car
        for car_id in range(self.num_cars):
            test_car = self.environment.test_cars[car_id]
            
            if not test_car.active:
                continue
            
            # Get state and action from AI
            state = self.environment.get_state(car_id)
            if state is None:
                continue
            
            action = self.agent.get_action(state, training=False)
            
            # Execute action
            self.environment.step(car_id, action)
        
        # Check if test is complete
        if self.environment.is_test_complete():
            self.test_complete = True
            self.test_running = False
            self._print_summary()
    
    def _print_summary(self):
        """Print test summary to console"""
        summary = self.environment.get_summary()
        
        print("\n" + "="*80)
        print("TEST COMPLETE - SUMMARY")
        print("="*80)
        print(f"Total Cars: {summary['total_cars']}")
        print(f"Finished: {summary['finished']} ({summary['finish_rate']}%)")
        print(f"Crashed: {summary['crashed']} ({summary['crash_rate']}%)")
        print(f"Timeout: {summary['timeout']} ({summary['timeout_rate']}%)")
        
        if summary['fastest_time'] > 0:
            print(f"\nFastest Completion:")
            print(f"  Car #{summary['fastest_car_id']}")
            print(f"  Time: {summary['fastest_time']:.2f}s")
        
        print("="*80 + "\n")
    
    def _draw_start_message(self):
        """Draw start message overlay"""
        if self.test_complete:
            return
        
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.display.blit(overlay, (0, 0))
        
        # Title
        title = self.font.render("AI Performance Test", True, (255, 215, 0))
        title_rect = title.get_rect(center=(WIDTH//2, HEIGHT//2 - 100))
        self.display.blit(title, title_rect)
        
        # Info
        info_lines = [
            f"{self.num_cars} AI cars will race simultaneously",
            "Each car has its own obstacles",
            "Pure exploitation mode (epsilon = 0.0)",
            "",
            "Press SPACE to start"
        ]
        
        y = HEIGHT//2 - 20
        for line in info_lines:
            text = self.small_font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(WIDTH//2, y))
            self.display.blit(text, text_rect)
            y += 35
    
    def _return_to_menu(self):
        """Return to main menu"""
        print("\nReturning to menu...")
        self.initialized = False
        self.test_running = False
        self.test_complete = False
        game_state_manager.setState('menu')
    
    def _save_and_exit(self):
        """Exit the game"""
        print("\nExiting...")
        pygame.quit()
        sys.exit(0)