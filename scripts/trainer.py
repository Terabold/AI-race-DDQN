# trainer.py - WITH WANDB INTEGRATION
import os
import pygame
import sys
import time
import numpy as np
import torch
import wandb

from scripts.AIEnvironment import AIEnvironment
from scripts.Constants import *
from scripts.dqn_agent import DQNAgent
from scripts.TrainingUtils import calculate_reward#, save_training_stats
from scripts.GameManager import game_state_manager

STATE_DIM = 33
ACTION_DIM = 9

# WandB Configuration
WANDB_PROJECT = "Racing-DQN-Training"
WANDB_RUN_ID = "gfb76"  # Consistent ID for resuming runs
WANDB_ENTITY = None  # Set to your wandb username if needed


class Trainer:
    
    def __init__(self, display, clock):
        self.display = display
        self.clock = clock
        self.environment = None
        self.agent = None
        self.initialized = False
        self.wandb_enabled = True  # Set to False to disable WandB
        
        # Training state
        self.steps = 0
        self.episode_reward = 0.0
        self.episode_reward_breakdown = {}  # Track detailed rewards
        self.losses = []
        self.state = None
        
        # Timing
        self.start_time = None
        self.fps_counter = 0
        self.fps_timer = None
        self.current_fps = 0
        
        # Statistics
        from collections import deque
        self.last_100_rewards = deque(maxlen=100)
        self.last_100_checkpoints = deque(maxlen=100)
        self.last_100_finishes = deque(maxlen=100)
        self.last_100_finish_times = deque(maxlen=100)
        self.last_100_losses = deque(maxlen=100)
        
        # Visualization
        self.show_visualization = False
        self.font = pygame.font.Font(FONT, 24)
        
    def initialize(self):
        """Initialize training with WandB"""
        if self.initialized:
            return
            
        print("\n" + "="*80)
        print("TRAINING - WITH WANDB LOGGING")
        print("="*80)
        
        # Disable audio
        pygame.mixer.quit()
        
        # Create environment
        self.environment = AIEnvironment(self.display)
        
        # Setup agent
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.agent = DQNAgent(device=device)
        
        # Check if resuming training
        model_exists = os.path.exists(self.agent.model_path)
        resume_training = model_exists
        
        # Load model if exists
        if model_exists:
            if self.agent.load_model(self.agent.model_path):
                print(f"Resuming from episode {self.agent.episode_count}")
        else:
            os.makedirs(self.agent.model_dir, exist_ok=True)
            self.agent.save_model()
            print("New model created")
        
        # Initialize WandB
        if self.wandb_enabled:
            self._init_wandb(resume_training)
        
        # Timing
        self.start_time = time.time()
        self.fps_timer = time.time()
        
        print(f"Episode: {self.agent.episode_count}")
        print("ESC=exit | V=toggle viz | W=toggle wandb logging\n")
        
        self._start_new_episode()
        self.initialized = True
    
    def _init_wandb(self, resume_training):
        """Initialize WandB with consistent run ID"""
        try:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                id=WANDB_RUN_ID,
                resume="allow" if resume_training else None,
                config={
                    "name": "Racing DQN Training",
                    "state_dim": STATE_DIM,
                    "action_dim": ACTION_DIM,
                    "learning_rate": self.agent.lr,
                    "gamma": self.agent.gamma,
                    "epsilon_start": 1.0,
                    "epsilon_min": self.agent.epsilon_min,
                    "epsilon_decay": self.agent.epsilon_decay,
                    "batch_size": self.agent.batch_size,
                    "replay_buffer_size": self.agent.replay_buffer.capacity,
                    "target_update_freq": self.agent.target_update,
                    "device": str(self.agent.device),
                    "network_hidden_dim": 256,
                    "optimizer": "Adam",
                    "loss_function": "Huber Loss",
                    "total_checkpoints": 25,
                    "target_time": 25.0,
                }
            )
            print(f"✓ WandB initialized: {WANDB_PROJECT}/{WANDB_RUN_ID}")
            
            # Define custom metrics
            wandb.define_metric("episode")
            
            # Episode metrics
            wandb.define_metric("episode/*", step_metric="episode")
            wandb.define_metric("performance/*", step_metric="episode")
            wandb.define_metric("rewards/*", step_metric="episode")
            
        except Exception as e:
            print(f"⚠ WandB init failed: {e}")
            print("Continuing without WandB logging...")
            self.wandb_enabled = False
    
    def _start_new_episode(self):
        """Start new episode"""
        self.environment.reset()
        self.steps = 0
        self.episode_reward = 0.0
        self.episode_reward_breakdown = {}
        self.losses.clear()
        self.state = self.environment.get_state()
    
    def _log_to_wandb(self, episode_data):
        """Log episode data to WandB"""
        if not self.wandb_enabled:
            return
        
        try:
            wandb.log(episode_data)
        except Exception as e:
            print(f"⚠ WandB logging error: {e}")
    
    def _end_episode(self):
        """End episode with WandB logging"""
        cp = self.environment.checkpoint_manager.crossed_count
        fin = self.environment.car_finished
        time_left = self.environment.time_remaining if fin else 0.0
        
        # Update agent
        self.agent.end_episode(
            episode_reward=self.episode_reward,
            checkpoints_reached=cp,
            time_remaining=time_left,
            finished=fin
        )
        
        # Track stats
        self.last_100_rewards.append(self.episode_reward)
        self.last_100_checkpoints.append(cp)
        self.last_100_finishes.append(fin)
        if fin:
            self.last_100_finish_times.append(time_left)
        if self.losses:
            self.last_100_losses.append(float(np.mean(self.losses)))
        
        # Calculate averages
        avg_reward_100 = float(np.mean(self.last_100_rewards)) if self.last_100_rewards else 0.0
        avg_checkpoints_100 = float(np.mean(self.last_100_checkpoints)) if self.last_100_checkpoints else 0.0
        avg_loss_100 = float(np.mean(self.last_100_losses)) if self.last_100_losses else 0.0
        win_rate_100 = float(sum(self.last_100_finishes)) / max(1, len(self.last_100_finishes)) * 100
        avg_finish_time_100 = float(np.mean(self.last_100_finish_times)) if self.last_100_finish_times else 0.0
        
        # Status
        if fin:
            comp = 25.0 - time_left
            status = f"🏁 FINISH ({comp:.2f}s)"
        elif self.environment.car_crashed:
            status = "💥 CRASH"
        elif self.environment.car_timeout:
            status = "⏱️ TIMEOUT"
        else:
            status = "UNKNOWN"
        
        # Console output
        avg_loss = float(np.mean(self.losses)) if self.losses else 0.0
        print(f"Ep {self.agent.episode_count:5d} | {status:20s} | "
              f"CP: {cp:2d}/25 | R: {self.episode_reward:7.1f} | "
              f"Loss: {avg_loss:.4f} | ε: {self.agent.epsilon:.4f} | "
              f"FPS: {self.current_fps:.0f}")
        
        # WandB logging - COMPREHENSIVE
        if self.wandb_enabled:
            episode_data = {
                "episode": self.agent.episode_count,
                "training/step": self.agent.train_step,
                
                # Episode metrics
                "episode/reward": self.episode_reward,
                "episode/checkpoints": cp,
                "episode/finished": 1 if fin else 0,
                "episode/crashed": 1 if self.environment.car_crashed else 0,
                "episode/timeout": 1 if self.environment.car_timeout else 0,
                "episode/steps": self.steps,
                "episode/avg_loss": avg_loss,
                
                # Performance metrics (rolling averages)
                "performance/avg_reward_100": avg_reward_100,
                "performance/avg_checkpoints_100": avg_checkpoints_100,
                "performance/avg_loss_100": avg_loss_100,
                "performance/win_rate_100": win_rate_100,
                "performance/checkpoint_progress_%": (avg_checkpoints_100 / 25) * 100,
                
                # Training metrics
                "training/epsilon": self.agent.epsilon,
                "training/learning_rate": self.agent.optimizer.param_groups[0]['lr'],
                "training/replay_buffer_size": len(self.agent.replay_buffer),
                "training/fps": self.current_fps,
                
                # Best metrics
                "performance/best_reward": self.agent.best_reward,
            }
            
            # Add finish time metrics if finished
            if fin:
                episode_data["episode/finish_time_remaining"] = time_left
                episode_data["episode/completion_time"] = 25.0 - time_left
                
            if self.last_100_finish_times:
                episode_data["performance/avg_finish_time_100"] = avg_finish_time_100
                episode_data["performance/avg_completion_time_100"] = 25.0 - avg_finish_time_100
            
            # Best finish time
            if self.agent.best_finish_time > 0:
                episode_data["performance/best_finish_time"] = self.agent.best_finish_time
                episode_data["performance/best_completion_time"] = 25.0 - self.agent.best_finish_time
            
            # Detailed reward breakdown
            if self.episode_reward_breakdown:
                for key, value in self.episode_reward_breakdown.items():
                    episode_data[f"rewards/{key}"] = float(value)
            
            self._log_to_wandb(episode_data)
        
        # Saves
        if self.agent.episode_count % 50 == 0:
            self.agent.save_model()
        
        # Milestones
        if self.agent.episode_count % 100 == 0:
            
            # save_training_stats(
            #     self.agent.episode_count, avg_reward_100, self.losses, self.agent.epsilon,
            #     win_count=wins, avg_checkpoints=avg_checkpoints_100, total_checkpoints=25,
            #     best_finish_time=self.agent.best_finish_time,
            #     best_finish_episode=self.agent.best_finish_episode,
            #     avg_finish_time=avg_finish_time_100
            # )
            
            elapsed = time.time() - self.start_time
            print("\n" + "="*80)
            print(f"MILESTONE - Episode {self.agent.episode_count}")
            print(f"  Wins: {win_rate_100}/100 ({win_rate_100:.1f}%) | Avg CP: {avg_checkpoints_100:.1f}/25")
            print(f"  Avg Reward: {avg_reward_100:.1f} | Avg Loss: {avg_loss_100:.4f}")
            if avg_finish_time_100 > 0:
                print(f"  Avg Completion Time: {25.0 - avg_finish_time_100:.2f}s")
            if self.agent.best_finish_time > 0:
                print(f"  Best Time: {25.0 - self.agent.best_finish_time:.2f}s (Ep {self.agent.best_finish_episode})")
            print(f"  Training Time: {elapsed/60:.1f}m | FPS: {self.current_fps:.0f}")
            print(f"  ε: {self.agent.epsilon:.4f} | LR: {self.agent.optimizer.param_groups[0]['lr']:.6f}")
            print("="*80 + "\n")
        
        self._start_new_episode()
    
    def run(self, dt):
        if not self.initialized:
            self.initialize()
        
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._save_and_exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._return_to_menu()
                elif event.key == pygame.K_v:
                    self.show_visualization = not self.show_visualization
                    print(f"Visualization: {'ON' if self.show_visualization else 'OFF'}")
        
        # Training step
        if not self.environment.episode_ended:
            action = self.agent.get_action(self.state, training=True)
            next_state, step_info, done = self.environment.step(action)
            
            # Calculate reward with breakdown
            reward, breakdown = calculate_reward(
                environment=self.environment,
                step_info=step_info,
                prev_state=self.state
            )
            
            # Accumulate reward breakdown for this episode
            for key, value in breakdown.items():
                if key not in self.episode_reward_breakdown:
                    self.episode_reward_breakdown[key] = 0.0
                self.episode_reward_breakdown[key] += float(value)
            
            self.agent.replay_buffer.add(self.state, action, reward, next_state, done)
            loss = self.agent.update()
            if loss is not None:
                self.losses.append(loss)
            
            self.steps += 1
            self.episode_reward += reward
            self.state = next_state
            self.fps_counter += 1
            
            # FPS calculation
            current_time = time.time()
            if current_time - self.fps_timer >= 1.0:
                self.current_fps = self.fps_counter / (current_time - self.fps_timer)
                self.fps_counter = 0
                self.fps_timer = current_time
        else:
            self._end_episode()
        
        # Rendering
        if self.show_visualization:
            self.environment.draw()
            pygame.display.update()
        else:        
            self.display.fill((0, 0, 0))
            
            lines = [
                f"Episode: {self.agent.episode_count}",
                f"V=viz"
            ]
            
            if self.agent.best_finish_time > 0:
                lines.append(f"Best: {25.0 - self.agent.best_finish_time:.2f}s (Ep {self.agent.best_finish_episode})")
            
            for i, line in enumerate(lines):
                t = self.font.render(line, True, (255, 255, 255))
                self.display.blit(t, (10, 10 + i * 30))
            
            pygame.display.update()
    
    def _return_to_menu(self):
        print("\nReturning to menu...")
        self.agent.save_model()
        print("Model saved")
        
        if self.wandb_enabled:
            try:
                wandb.finish()
                print("WandB run finished")
            except:
                pass
        
        try:
            pygame.mixer.init()
        except:
            pass
        self.initialized = False
        game_state_manager.setState('menu')
    
    def _save_and_exit(self):
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
                print("WandB run finished")
            except:
                pass
        
        pygame.quit()
        sys.exit(0)