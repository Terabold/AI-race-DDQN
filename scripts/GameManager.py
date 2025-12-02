# tracks game state and settings
class GameStateManager:
    def __init__(self):
        self.state = 'menu'
        
        self.player1_selection = None
        self.player2_selection = None
        self.player1_car_color = "Blue"
        self.player2_car_color = "Red"
        
        self.tester_num_cars = 10
        
    def setState(self, new_state):
        self.state = new_state
        
    def getState(self):
        return self.state

game_state_manager = GameStateManager()