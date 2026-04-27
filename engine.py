# לולאה ראשית - מחליפה בין מסכים
import pygame
from scripts.Constants import FPS, MENUBG, WIDTH, HEIGHT
from scripts.menu import MainMenu, RaceSettingsMenu
from scripts.GameManager import game_state_manager
from scripts.ResourceLoader import resource_manager

class Engine:
    def __init__(self):
        # התחלת pygame
        pygame.init()
        pygame.joystick.quit()
        
        # יצירת חלון משחק
        pygame.display.set_caption('DDQN RACE')
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        
        # מסך שחור
        self.display.fill((0, 0, 0))
        pygame.display.flip()
        
        # להתחיל טעינת משאבים
        resource_manager.initialize()
        
        # להראות מסך טעינה זמני
        font = pygame.font.Font(None, 48)
        loading_messages = ["Loading.", "Loading..", "Loading..."]
        
        # טעינת הכל
        resource_manager.load_all()
        
        # אנימציית טעינה קוסמטית (לשיפור חווית המשתמש ומעבר חלק לתפריט)
        for msg in loading_messages * 2:
            self.display.fill((0, 0, 0))
            text = font.render(msg, True, (255, 255, 255))
            self.display.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            pygame.display.flip()
            pygame.time.wait(200)
        
        # רקע תפריט
        self.menu_bg = resource_manager.images['menu_bg']
        
        self.game = None
        self.trainer = None
        
        self.main_menu = MainMenu(self.display)
        self.settings_menu = RaceSettingsMenu(self.display)
        
    def run(self):
        previous_state = None

        while True:
            # קבלת המצב הנוכחי (תפריט, משחק, או אימון) מנהל המצבים
            current_state = game_state_manager.get_state()
            
            # בדיקת אירועי יציאה מהמשחק
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # ניהול מעברים בין מצבי המשחק
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

            # הגדרת הגבלת פריימים למשחק ולאימון
            if current_state == 'training':
                self.clock.tick()
            else:
                self.clock.tick(FPS)

            # להריץ את המסך של המצב משחק הנבחר
            if current_state == 'menu':
                self.display.blit(self.menu_bg, (0, 0))
                self.main_menu.run()
            elif current_state == 'settings':
                self.display.blit(self.menu_bg, (0, 0))
                self.settings_menu.run()
            elif current_state == 'game':
                self.game.run()
            elif current_state == 'training':
                self.trainer.run()

            previous_state = current_state
            pygame.display.flip()


if __name__ == '__main__':
    Engine().run()