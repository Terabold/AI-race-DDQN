import pygame
import torch
from scripts.TesterGame import TesterGame
from scripts.GameManager import game_state_manager
from scripts.Constants import FPS

class TesterEnvironment:
    """Wrapper to run tester through the main game engine"""
    
    def __init__(self, display, clock):
        self.display = display
        self.clock = clock
        self.tester_game = TesterGame(display, clock)
    
    def run(self, dt):
        """Called by engine.py in training loop"""
        if game_state_manager.getState() != 'tester':
            return
        
        self.tester_game.run(dt)
        pygame.display.flip()