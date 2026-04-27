# פצצות - מכשולים אקראיים שמאטים את הרכב
import pygame
from scripts.Constants import BOMB, OBSTACLE_SIZE, OBSTACLE_HITBOX, BOMB_LIST
from scripts.ResourceLoader import resource_manager
import random

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, show_image=True):
        super().__init__()
        self.show_image = show_image
        if show_image:
            self.image = resource_manager.images['bomb']
            self.image = pygame.transform.scale(self.image, OBSTACLE_SIZE)
        else:
            # רק קופסא אדומה לאימון בלי תמונה
            self.image = pygame.Surface(OBSTACLE_SIZE, pygame.SRCALPHA)
            pygame.draw.rect(self.image, (255, 0, 0), OBSTACLE_HITBOX)
        
        # תחומי גודל להתנגשות
        self.mask_surface = pygame.Surface(OBSTACLE_SIZE, pygame.SRCALPHA) 
        pygame.draw.rect(self.mask_surface, (255, 255, 255), OBSTACLE_HITBOX)
        self.mask = pygame.mask.from_surface(self.mask_surface)  # מסכה לזיהוי התנגשות
        self.rect = self.image.get_rect(center=(x, y))

    def generate_obstacles(self, num_obstacles=10):  
        obstacle_group = pygame.sprite.Group()
        available_positions = BOMB_LIST  
        random.shuffle(available_positions)
        selected_positions = available_positions[:num_obstacles]
        for x, y in selected_positions:
            obstacle = Obstacle(x, y, self.show_image)
            obstacle_group.add(obstacle)
        return obstacle_group

    def reshuffle_obstacles(self, obstacle_group, num_obstacles):
        obstacle_group.empty()  
        obstacle_group.add(self.generate_obstacles(num_obstacles))