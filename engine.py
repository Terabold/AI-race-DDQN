# main loop - switches between screens
import pygame
from scripts.Constants import FPS, MENUBG, WIDTH, HEIGHT
from scripts.menu import MainMenu, RaceSettingsMenu
from scripts.GameManager import game_state_manager
from scripts.ResourceLoader import resource_manager

class Engine:
    def __init__(self):
        # Initialize pygame ONLY
        pygame.init()
        pygame.joystick.quit()
        
        # Create window IMMEDIATELY - before any heavy operations
        pygame.display.set_caption('DDQN RACE')
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Show black screen instantly
        self.display.fill((0, 0, 0))
        pygame.display.flip()
        
        # Initialize device
        resource_manager.initialize()
        
        # Show loading text IMMEDIATELY, load in background
        font = pygame.font.Font(None, 48)
        loading_messages = ["Loading.", "Loading..", "Loading..."]
        
        # First frame - load everything
        resource_manager.load_all()
        
        # Show animation while resources loaded
        for msg in loading_messages * 2:
            self.display.fill((0, 0, 0))
            text = font.render(msg, True, (255, 255, 255))
            self.display.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            pygame.display.flip()
            pygame.time.wait(200)
        
        # Get loaded background
        self.menu_bg = resource_manager.images['menu_bg']
        
        self.game = None
        self.trainer = None
        
        self.main_menu = MainMenu(self.display)
        self.settings_menu = RaceSettingsMenu(self.display)
        
    def run(self):
        previous_state = None

        while True:
            current_state = game_state_manager.getState()
            
            # init when switching states
            if previous_state != current_state:
                if current_state == 'game':
                    if self.game is None:
                        from scripts.Game import Game
                        self.game = Game(self.display, self.clock)
                    self.game.initialize_environment()
                elif current_state == 'training':
                    if self.trainer is None:
                        from scripts.trainer import Trainer
                        self.trainer = Trainer(self.display, self.clock)
                    self.trainer.initialize()

            # unlimited fps for training, 60 for menus
            if current_state == 'training':
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
            elif current_state == 'game':
                self.game.run(dt)
            elif current_state == 'training':
                self.trainer.run(dt)

            previous_state = current_state
            pygame.display.flip()


if __name__ == '__main__':
    Engine().run()