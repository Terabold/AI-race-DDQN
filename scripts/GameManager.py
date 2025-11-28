"""
GAMEMANAGER.PY - Tracks what screen/mode the game is in
Like a traffic controller directing which part of the game is active
"""

class GameStateManager:
    def __init__(self):
        # Current screen: 'menu', 'settings', 'tester_settings', 'game', 'training', 'tester'
        self.state = 'menu'
        self.previous_state = None
        
        # Player settings (set in race settings menu)
        self.player1_selection = None      # "Human" or "DQN" or None
        self.player2_selection = None
        self.player1_car_color = "Blue"
        self.player2_car_color = "Red"
        
        # Tester settings
        self.tester_num_cars = 10
        
    def setState(self, new_state):
        """Switch to a different screen"""
        self.previous_state = self.state
        self.state = new_state
        
    def getState(self):
        return self.state

# Single global instance used everywhere
game_state_manager = GameStateManager()