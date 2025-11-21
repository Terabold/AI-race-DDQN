# main.py - Updated with TesterGame integration
import pygame
import sys
from scripts.Constants import WIDTH, HEIGHT, FPS, DISPLAY_SIZE
from scripts.GameManager import game_state_manager
from scripts.menu import Menu
from scripts.Game import Game
from scripts.trainer import Trainer
from scripts.TesterGame import TesterGame  # NEW

def main():
    pygame.init()
    
    # Setup display
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI Racing Game")
    clock = pygame.time.Clock()
    
    # Initialize all game modes
    menu = Menu(display, clock)
    game = Game(display, clock)
    trainer = Trainer(display, clock)
    tester = TesterGame(display, clock)  # NEW
    
    # Main game loop
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        current_state = game_state_manager.getState()
        
        if current_state == 'menu':
            menu.run()
        elif current_state == 'game':
            game.run(dt)
        elif current_state == 'training':
            trainer.run(dt)
        elif current_state == 'tester':  # NEW
            tester.run(dt)
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()