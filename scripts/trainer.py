# trainer.py - SIMPLIFIED & CLEAN
import os
import pygame
import sys
import time
import numpy as np
import torch
import wandb
from collections import deque

from scripts.AIEnvironment import AIEnvironment
from scripts.Constants import *
from scripts.dqn_agent import DQNAgent
from scripts.Reward import calculate_reward
from scripts.GameManager import game_state_manager

STATE_DIM = 33
ACTION_DIM = 9

# WandB Configuration
WANDB_PROJECT = "Racing-DQN"
WANDB_RUN_ID = "train_3.0"

class Trainer: 
    def __init__(self, display, clock):
        self.display = display
        self.clock = clock
        self.environment = None
        self.agent = None
        self.initialized = False
        self.wandb_enabled = True
        
        # Training state
        self.steps = 0
        self.episode_reward = 0.0
        self.losses = []
        self.state = None
        
        # Event tracking for this episode
        self.episode_events = {
            'checkpoint_crosses': 0,
            'backward_crosses': 0,
            'obstacle_hits': 0,
            'wall_crashes': 0,
            'timeouts': 0
        }
        
        # Timing
        self.start_time = None
        self.fps_counter = 0
        self.fps_timer = None
        self.current_fps = 0
        
        # Rolling stats (last 100 episodes)
        self.last_100_rewards = deque(maxlen=100)
        self.last_100_checkpoints = deque(maxlen=100)
        self.last_100_losses = deque(maxlen=100)
        self.last_100_outcomes = deque(maxlen=100)  # 0=crash, 1=timeout, 2=finish
        self.last_100_completion_times = deque(maxlen=100)  # Only for finishes
        self.last_100_obstacle_hits = deque(maxlen=100)
        
        # Visualization
        self.show_visualization = False
        self.font = pygame.font.Font(FONT, 24)
        
    def initialize(self):
        if self.initialized:
            return
            
        print("\n" + "="*80)
        print("TRAINING MODE")
        print("="*80)
        
        pygame.mixer.quit()
        
        self.environment = AIEnvironment(self.display)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.agent = DQNAgent(device=device)
        
        # Load or create model
        model_exists = os.path.exists(self.agent.model_path)
        if model_exists:
            if self.agent.load_model(self.agent.model_path):
                print(f"Resuming from episode {self.agent.episode_count}")
        else:
            os.makedirs(self.agent.model_dir, exist_ok=True)
            self.agent.save_model()
            print("New model created")
        
        # Initialize WandB
        if self.wandb_enabled:
            self.init_wandb(model_exists)
        
        self.start_time = time.time()
        self.fps_timer = time.time()
        
        print(f"Episode: {self.agent.episode_count}")
        print("ESC=menu | V=toggle viz\n")
        
        self.start_new_episode()
        self.initialized = True
    
    def init_wandb(self, resume_training):
        try:
            wandb.init(
                project=WANDB_PROJECT,
                entity=None,
                id=WANDB_RUN_ID,
                resume="allow" if resume_training else None,
                config={
                    "state_dim": STATE_DIM,
                    "action_dim": ACTION_DIM,
                    "learning_rate": self.agent.lr,
                    "gamma": self.agent.gamma,
                    "epsilon_min": self.agent.epsilon_min,
                    "batch_size": self.agent.batch_size,
                    "buffer_size": self.agent.replay_buffer.capacity,
                }
            )
            print(f"✓ WandB: {WANDB_PROJECT}/{WANDB_RUN_ID}")
            
            wandb.define_metric("episode")
            wandb.define_metric("episode/*", step_metric="episode")
            wandb.define_metric("performance/*", step_metric="episode")
            
        except Exception as e:
            print(f"⚠ WandB init failed: {e}")
            self.wandb_enabled = False
    
    def start_new_episode(self):
        self.environment.reset()
        self.steps = 0
        self.episode_reward = 0.0
        self.losses.clear()
        self.state = self.environment.get_state()
        self.episode_events = {
            'checkpoint_crosses': 0,
            'backward_crosses': 0,
            'obstacle_hits': 0,
            'wall_crashes': 0,
            'timeouts': 0
        }
    
    def end_episode(self):
        cp = self.environment.checkpoint_manager.crossed_count
        finished = self.environment.car_finished
        crashed = self.environment.car_crashed
        timeout = self.environment.car_timeout
        
        # Determine outcome
        if finished:
            outcome = 2
            completion_time = 25.0 - self.environment.time_remaining
            self.last_100_completion_times.append(completion_time)
        elif timeout:
            outcome = 1
            completion_time = 25.0
        else:  # crashed
            outcome = 0
            completion_time = 25.0
        
        # Update agent
        self.agent.end_episode(
            episode_reward=self.episode_reward,
            checkpoints_reached=cp,
            time_remaining=self.environment.time_remaining if finished else 0.0,
            finished=finished
        )
        
        # Track stats
        self.last_100_rewards.append(self.episode_reward)
        self.last_100_checkpoints.append(cp)
        self.last_100_outcomes.append(outcome)
        self.last_100_obstacle_hits.append(self.episode_events['obstacle_hits'])
        if self.losses:
            self.last_100_losses.append(float(np.mean(self.losses)))
        
        # Calculate averages
        avg_reward = float(np.mean(self.last_100_rewards)) if self.last_100_rewards else 0.0
        avg_checkpoints = float(np.mean(self.last_100_checkpoints)) if self.last_100_checkpoints else 0.0
        avg_loss = float(np.mean(self.last_100_losses)) if self.last_100_losses else 0.0
        avg_obstacles = float(np.mean(self.last_100_obstacle_hits)) if self.last_100_obstacle_hits else 0.0
        
        # Win rate (only finishes count as wins)
        win_rate = (sum(1 for x in self.last_100_outcomes if x == 2) / max(1, len(self.last_100_outcomes))) * 100
        
        # Average completion time (only for finishes)
        avg_completion = float(np.mean(self.last_100_completion_times)) if self.last_100_completion_times else 0.0
        
        # Status
        if finished:
            status = f"✓ FINISH ({completion_time:.2f}s)"
        elif crashed:
            status = "✗ CRASH"
        else:
            status = "⏱ TIMEOUT"
        
        # Console
        print(f"Ep {self.agent.episode_count:5d} | {status:20s} | "
              f"CP:{cp:2d} | R:{self.episode_reward:6.1f} | "
              f"ε:{self.agent.epsilon:.3f} | FPS:{self.current_fps:.0f}")
        
        # WandB - ONLY WHAT MATTERS
        if self.wandb_enabled:
            log_data = {
                "episode": self.agent.episode_count,
                
                # Current episode
                "episode/reward": self.episode_reward,
                "episode/checkpoints": cp,
                "episode/outcome": outcome,  # 0=crash, 1=timeout, 2=finish
                "episode/loss": float(np.mean(self.losses)) if self.losses else 0.0,
                
                # Performance (rolling 100)
                "performance/win_rate": win_rate,
                "performance/avg_reward": avg_reward,
                "performance/avg_checkpoints": avg_checkpoints,
                "performance/avg_loss": avg_loss,
                "performance/avg_obstacle_hits": avg_obstacles,
                
                # Training state
                "training/epsilon": self.agent.epsilon,
                "training/learning_rate": self.agent.optimizer.param_groups[0]['lr'],
                "training/buffer_size": len(self.agent.replay_buffer),
            }
            
            # Add completion time only if finished
            if finished:
                log_data["episode/completion_time"] = completion_time
            
            # Add average completion time if we have finishes
            if self.last_100_completion_times:
                log_data["performance/avg_completion_time"] = avg_completion
            
            # Best time
            if self.agent.best_finish_time > 0:
                log_data["performance/best_completion_time"] = 25.0 - self.agent.best_finish_time
            
            try:
                wandb.log(log_data)
            except:
                pass
        
        # Auto-save
        if self.agent.episode_count % 50 == 0:
            self.agent.save_model()
        
        # Milestones
        if self.agent.episode_count % 100 == 0:
            elapsed = time.time() - self.start_time
            print("\n" + "="*80)
            print(f"MILESTONE - Episode {self.agent.episode_count}")
            print(f"  Win Rate: {win_rate:.1f}% | Avg CP: {avg_checkpoints:.1f}/25")
            print(f"  Avg Reward: {avg_reward:.1f} | Avg Loss: {avg_loss:.4f}")
            print(f"  Avg Obstacles Hit: {avg_obstacles:.1f}")
            if avg_completion > 0:
                print(f"  Avg Completion: {avg_completion:.2f}s")
            if self.agent.best_finish_time > 0:
                print(f"  Best Time: {25.0 - self.agent.best_finish_time:.2f}s (Ep {self.agent.best_finish_episode})")
            print(f"  Time: {elapsed/60:.1f}m | ε: {self.agent.epsilon:.4f}")
            print("="*80 + "\n")
        
        self.start_new_episode()
    
    def run(self, dt):
        if not self.initialized:
            self.initialize()
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_and_exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.return_to_menu()
                elif event.key == pygame.K_v:
                    self.show_visualization = not self.show_visualization
        
        # Training step
        if not self.environment.episode_ended:
            action = self.agent.get_action(self.state, training=True)
            next_state, step_info, done = self.environment.step(action)
            
            # Track events
            if step_info.get('checkpoint_crossed'):
                self.episode_events['checkpoint_crosses'] += 1
            if step_info.get('backward_crossed'):
                self.episode_events['backward_crosses'] += 1
            if step_info.get('hit_obstacle'):
                self.episode_events['obstacle_hits'] += 1
            if step_info.get('collision'):
                self.episode_events['wall_crashes'] += 1
            if step_info.get('timeout'):
                self.episode_events['timeouts'] += 1
            
            reward, _ = calculate_reward(self.environment, step_info, self.state)
            
            self.agent.replay_buffer.add(self.state, action, reward, next_state, done)
            loss = self.agent.update()
            if loss is not None:
                self.losses.append(loss)
            
            self.steps += 1
            self.episode_reward += reward
            self.state = next_state
            self.fps_counter += 1
            
            # FPS
            current_time = time.time()
            if current_time - self.fps_timer >= 1.0:
                self.current_fps = self.fps_counter / (current_time - self.fps_timer)
                self.fps_counter = 0
                self.fps_timer = current_time
        else:
            self.end_episode()
        
        # Rendering
        if self.show_visualization:
            self.environment.draw()
        else:
            self.display.fill((0, 0, 0))
            lines = [
                f"Episode: {self.agent.episode_count}",
                f"V=viz | ESC=menu"
            ]
            if self.agent.best_finish_time > 0:
                lines.append(f"Best: {25.0 - self.agent.best_finish_time:.2f}s")
            
            for i, line in enumerate(lines):
                t = self.font.render(line, True, (255, 255, 255))
                self.display.blit(t, (10, 10 + i * 30))
        
        pygame.display.update()
    
    def return_to_menu(self):
        print("\nReturning to menu...")
        self.agent.save_model()
        if self.wandb_enabled:
            try:
                wandb.finish()
            except:
                pass
        pygame.mixer.init()

        self.initialized = False
        game_state_manager.setState('menu')
    
    def save_and_exit(self):
        print("\nSaving and exiting...")
        if self.agent:
            self.agent.save_model()
            elapsed = time.time() - self.start_time
            print(f"Episodes: {self.agent.episode_count} | Time: {elapsed/60:.1f}m")
            if self.agent.best_finish_time > 0:
                print(f"Best Time: {25.0 - self.agent.best_finish_time:.2f}s")
        
        if self.wandb_enabled:
            try:
                wandb.finish()
            except:
                pass
        
        pygame.quit()
        sys.exit(0)