# handles actual racing - human vs ai etc
import pygame
import sys
import os
from scripts.Constants import INFERENCE_EPSILON
from scripts.Environment import Environment
from scripts.Human_Agent import HumanAgent  
from scripts.dqn_agent import DQNAgent
from scripts.GameManager import game_state_manager


class Game:
    def __init__(self, display, clock):
        self.display = display
        self.clock = clock
        self.environment = None
        self.player1 = None
        self.player2 = None

    def initialize_environment(self):
        settings = {
            'player1': game_state_manager.player1_selection,
            'player2': game_state_manager.player2_selection,
            'car_color1': game_state_manager.player1_car_color,
            'car_color2': game_state_manager.player2_car_color
        }

        self.environment = Environment(
            self.display,
            car_color1=settings['car_color1'] if settings['player1'] else None,
            car_color2=settings['car_color2'] if settings['player2'] else None
        )
        
        self.player1 = self.create_player(settings['player1'], 1)
        self.player2 = self.create_player(settings['player2'], 2)

    def create_player(self, player_type, player_num):
        if player_type == "Human":
            return HumanAgent(player_num)
        elif player_type == "DQN":
            agent = DQNAgent()
            if os.path.exists(agent.model_path):
                agent.load_model(agent.model_path)
            agent.epsilon = INFERENCE_EPSILON  # small randomness
            agent.policy_net.eval()
            agent.target_net.eval()
            return agent
        return None

    def run(self, dt):
        if game_state_manager.getState() != 'game':
            return
        
        if not self.environment:
            self.initialize_environment()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if self.environment.game_state == "paused":
                self.environment.pause_menu.handle_event(event, self.environment)
                continue
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.environment.game_state in ["finished", "failed"]:
                    self.environment.restart_game()
                elif event.key == pygame.K_r:
                    self.environment.restart_game()
                elif event.key == pygame.K_ESCAPE and self.environment.game_state == "running":
                    self.environment.toggle_pause()

        if self.environment.game_state != "paused":
            self.environment.update()
            if self.environment.game_state == "running":
                p1_action = self.get_action(self.player1, 1)
                p2_action = self.get_action(self.player2, 2)
                self.environment.move(p1_action, p2_action)

        self.environment.draw()

    def get_action(self, player, car_num):
        if player is None:
            return None
        if isinstance(player, DQNAgent):
            state = self.environment.get_state(car_num=car_num)
            return player.get_action(state, training=False) if state is not None else 0
        return player.get_action()