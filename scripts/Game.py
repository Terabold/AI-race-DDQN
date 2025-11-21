import pygame
import sys
from scripts.Constants import INFERENCE_EPSILON
from scripts.Environment import Environment
from scripts.Human_Agent import HumanAgentWASD, HumanAgentArrows
from scripts.dqn_agent import DQNAgent
from scripts.GameManager import game_state_manager
import os


class Game:
    def __init__(self, display, clock):
        self.display = display
        self.clock = clock
        self.environment = None
        self.player1 = None
        self.player2 = None

    def initialize_environment(self, settings=None):
        if settings is None:
            settings = {
                'player1': game_state_manager.player1_selection,
                'player2': game_state_manager.player2_selection,
                'car_color1': game_state_manager.player1_car_color,
                'car_color2': game_state_manager.player2_car_color
            }

        self.environment = Environment(
            self.display,
            car_color1=settings['car_color1'] if settings.get('player1') else None,
            car_color2=settings['car_color2'] if settings.get('player2') else None
        )
        
        self.player1 = self._create_player(settings.get('player1'), settings['car_color1'], 1)
        self.player2 = self._create_player(settings.get('player2'), settings['car_color2'], 2)

    def _create_player(self, player_type, car_color, player_num):
        """Create a single player (Human or AI)"""
        if player_type == "Human":
            agent = HumanAgentWASD() if player_num == 1 else HumanAgentArrows()
            controls = "WASD" if player_num == 1 else "Arrows"
            print(f"Player {player_num}: Human ({controls}) - {car_color} car")
            return agent
            
        elif player_type == "DQN":
            agent = DQNAgent()
            
            if os.path.exists(agent.model_path):
                agent.load_model(agent.model_path)
                print(f"Player {player_num}: AI loaded from {agent.model_path}")
            else:
                print(f"Warning: No trained model found for Player {player_num}!")
            
            agent.epsilon = INFERENCE_EPSILON
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
                elif event.key == pygame.K_r and self.environment.game_state in ["finished", "failed", "running"]:
                    self.environment.restart_game()
                elif event.key == pygame.K_ESCAPE and self.environment.game_state == "running":
                    self.environment.toggle_pause()

        if self.environment.game_state != "paused":
            self.environment.update()
            p1_action = self._get_player_action(self.player1, car_num=1)
            p2_action = self._get_player_action(self.player2, car_num=2)
            self.environment.move(p1_action, p2_action)

        self.environment.draw()

    def _get_player_action(self, player, car_num):
        if player is None:
            return None
        
        if isinstance(player, DQNAgent):
            state = self.environment.get_state(car_num=car_num)
            return player.get_action(state, training=False) if state is not None else 0
        
        return player.get_action()