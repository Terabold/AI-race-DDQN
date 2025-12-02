# main loop - switches between screens
import pygame
from scripts.Constants import FPS, MENUBG, WIDTH, HEIGHT
from scripts.Game import Game
from scripts.menu import MainMenu, RaceSettingsMenu, TesterSettingsMenu
from scripts.GameManager import game_state_manager
from scripts.trainer import Trainer
from scripts.TesterGame import TesterGame

class Engine:
    def __init__(self):
        pygame.init()
        pygame.joystick.quit()  # no controller
        pygame.display.set_caption('DDQN RACE')
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.menu_bg = pygame.transform.scale(
            pygame.image.load(MENUBG), (WIDTH, HEIGHT)
        )
        
        self.game = Game(self.display, self.clock)
        self.trainer = Trainer(self.display, self.clock)
        self.tester = TesterGame(self.display, self.clock)
        
        self.main_menu = MainMenu(self.display)
        self.settings_menu = RaceSettingsMenu(self.display)
        self.tester_menu = TesterSettingsMenu(self.display)
        
    def run(self):
        previous_state = None

        while True:
            current_state = game_state_manager.getState()
            
            # init when switching states
            if previous_state != current_state:
                if current_state == 'game':
                    self.game.initialize_environment()
                elif current_state == 'training':
                    self.trainer.initialize()
                elif current_state == 'tester':
                    self.tester.initialize()

            # unlimited fps for training/testing, 60 for menus
            if current_state in ['training', 'tester']:
                dt = self.clock.tick() / 1000.0
            else:
                dt = self.clock.tick(FPS) / 1000.0

            # run active screen
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
            pygame.display.flip()


if __name__ == '__main__':
    Engine().run()