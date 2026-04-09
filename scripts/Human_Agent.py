# קריאת קלט מקלדת
# WASD לשחקן 1, חצים לשחקן 2
# פעולות: 0=כלום, 1=קדימה, 2=אחורה, 3=שמאל, 4=ימין, 5-8=שילובים
import pygame


class HumanAgent:
    CONTROLS = {
        1: {'forward': pygame.K_w, 'backward': pygame.K_s, 'left': pygame.K_a, 'right': pygame.K_d},
        2: {'forward': pygame.K_UP, 'backward': pygame.K_DOWN, 'left': pygame.K_LEFT, 'right': pygame.K_RIGHT}
    }
    
    def __init__(self, player_num=1):
        self.controls = self.CONTROLS.get(player_num, self.CONTROLS[1])
    
    def get_action(self):
        keys = pygame.key.get_pressed()
        
        forward = keys[self.controls['forward']]
        backward = keys[self.controls['backward']]
        left = keys[self.controls['left']]
        right = keys[self.controls['right']]
        
        # שילובים קודם
        if forward and left:
            return 5
        if forward and right:
            return 6
        if backward and left:
            return 7
        if backward and right:
            return 8
        # יחידים
        if forward:
            return 1
        if backward:
            return 2
        if left:
            return 3
        if right:
            return 4
        
        return 0