"""
ENGINE.PY - The heart of the game
Controls which screen is shown and keeps the game running
Think of it as a TV remote that switches between channels
"""

import pygame
from scripts.Constants import DISPLAY_SIZE, FPS, FONT, MENUBG
from scripts.Game import Game
from scripts.menu import MainMenu, RaceSettingsMenu, TesterSettingsMenu
from scripts.GameManager import game_state_manager
from scripts.trainer import Trainer
from scripts.TesterGame import TesterGame

class Engine:
    def __init__(self):
        # Initialize pygame and create window
        pygame.init()
        pygame.joystick.quit()  # Disable controller input
        pygame.display.set_caption('AI Racing Game')
        self.display = pygame.display.set_mode((1600, 900))
        self.clock = pygame.time.Clock()
        
        # Load menu background image
        self.menu_bg = pygame.transform.scale(
            pygame.image.load(MENUBG), (1600, 900)
        )
        
        # Create all game screens (only one is active at a time)
        self.game = Game(self.display, self.clock)           # Human/AI racing
        self.trainer = Trainer(self.display, self.clock)      # AI training mode
        self.tester = TesterGame(self.display, self.clock)    # AI testing mode
        
        # Menu screens
        self.main_menu = MainMenu(self.display)
        self.settings_menu = RaceSettingsMenu(self.display)
        self.tester_menu = TesterSettingsMenu(self.display)
        
    def run(self):
        """Main loop - runs forever until game closes"""
        previous_state = None

        while True:
            current_state = game_state_manager.getState()
            
            # When switching states, initialize the new screen
            if previous_state != current_state:
                if current_state == 'game':
                    self.game.initialize_environment()
                elif current_state == 'training':
                    self.trainer.initialize()
                elif current_state == 'tester':
                    self.tester.initialize()

            # FPS: Unlimited for AI training/testing, 60 FPS for menus/gameplay
            if current_state in ['training', 'tester']:
                dt = self.clock.tick() / 1000.0
            else:
                dt = self.clock.tick(FPS) / 1000.0

            # Run the active screen
            if current_state == 'menu':
                self.display.blit(self.menu_bg, (0, 0))
                self.main_menu.run()
            elif current_state == 'settings':
                self.display.blit(self.menu_bg, (0, 0))
                self.settings_menu.run()
            elif current_state == 'tester_settings':
                self.display.blit(self.menu_bg, (0, 0))
                self.tester_menu.run()
            elif current_state == 'game':
                self.game.run(dt)
            elif current_state == 'training':
                self.trainer.run(dt)
            elif current_state == 'tester':
                self.tester.run(dt)

            previous_state = current_state
            pygame.display.flip()  # Update screen


if __name__ == '__main__':
    Engine().run()