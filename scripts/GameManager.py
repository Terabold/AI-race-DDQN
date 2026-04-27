class GameStateManager:  
    def __init__(self):
        # מצב משחק ברירת מחדל על התפריט
        self.state = 'menu'
        
        # הגדרות שחקנים
        # סוג שחקן
        self.player1_selection = None
        self.player2_selection = None

        # צבעים של המכוניות 
        self.player1_car_color = "Blue"
        self.player2_car_color = "Red"

    # להגדיר מצב משחק  
    def setState(self, new_state):
        self.state = new_state
    # לקבל מצב משחק  
    def getState(self):
        return self.state

# מופע גלובלי יחיד - משותף לכל חלקי המשחק
game_state_manager = GameStateManager()