# מנהל מצב המשחק הגלובלי והגדרות שחקנים

class GameStateManager:
    """
    Global state manager for game flow and player settings.
    
    States: 'menu' -> 'settings' -> 'game' or 'training'
    """
    
    def __init__(self):
        self.state = 'menu'
        
        # הגדרות שחקנים
        self.player1_selection = None
        self.player2_selection = None
        self.player1_car_color = "Blue"
        self.player2_car_color = "Red"
        
    def setState(self, new_state):
        self.state = new_state
        
    def getState(self):
        return self.state

# מופע גלובלי יחיד - משותף לכל חלקי המשחק
game_state_manager = GameStateManager()