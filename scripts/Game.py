"""
GAME.PY - Handles actual racing gameplay
Manages players (human or AI) and processes their inputs
"""

import pygame
import sys
import os
from scripts.Constants import INFERENCE_EPSILON
from scripts.Environment import Environment
from scripts.Human_Agent import HumanAgentWASD, HumanAgentArrows
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
        """Setup race with selected players and cars"""
        settings = {
            'player1': game_state_manager.player1_selection,
            'player2': game_state_manager.player2_selection,
            'car_color1': game_state_manager.player1_car_color,
            'car_color2': game_state_manager.player2_car_color
        }

        # Create race environment
        self.environment = Environment(
            self.display,
            car_color1=settings['car_color1'] if settings['player1'] else None,
            car_color2=settings['car_color2'] if settings['player2'] else None
        )
        
        # Create player controllers
        self.player1 = self._create_player(settings['player1'], 1)
        self.player2 = self._create_player(settings['player2'], 2)

    def _create_player(self, player_type, player_num):
        """Create human or AI controller"""
        if player_type == "Human":
            # Player 1 uses WASD, Player 2 uses Arrow keys
            return HumanAgentWASD() if player_num == 1 else HumanAgentArrows()
        elif player_type == "DQN":
            # Load trained AI
            agent = DQNAgent()
            if os.path.exists(agent.model_path):
                agent.load_model(agent.model_path)
            agent.epsilon = INFERENCE_EPSILON  # Minimal randomness
            agent.policy_net.eval()
            agent.target_net.eval()
            return agent
        return None

    def run(self, dt):
        """Called every frame during gameplay"""
        if game_state_manager.getState() != 'game':
            return
        
        if not self.environment:
            self.initialize_environment()

        # Handle keyboard/mouse events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Pause menu handles its own events
            if self.environment.game_state == "paused":
                self.environment.pause_menu.handle_event(event, self.environment)
                continue
            
            if event.type == pygame.KEYDOWN:
                # Restart race
                if event.key == pygame.K_SPACE and self.environment.game_state in ["finished", "failed"]:
                    self.environment.restart_game()
                elif event.key == pygame.K_r:
                    self.environment.restart_game()
                # Pause
                elif event.key == pygame.K_ESCAPE and self.environment.game_state == "running":
                    self.environment.toggle_pause()

        # Update game if not paused
        if self.environment.game_state != "paused":
            self.environment.update()
            p1_action = self._get_action(self.player1, 1)
            p2_action = self._get_action(self.player2, 2)
            self.environment.move(p1_action, p2_action)

        self.environment.draw()

    def _get_action(self, player, car_num):
        """Get action from player (human input or AI decision)"""
        if player is None:
            return None
        if isinstance(player, DQNAgent):
            state = self.environment.get_state(car_num=car_num)
            return player.get_action(state, training=False) if state is not None else 0
        return player.get_action()