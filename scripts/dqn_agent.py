"""
DQN_AGENT.PY - The AI trainer
Manages learning: stores experiences, updates the brain, balances exploration vs exploitation

KEY CONCEPTS:
- Epsilon: Chance of random action (exploration). Starts high, decreases over time.
- Replay Buffer: Memory of past experiences to learn from
- Target Network: Stable copy of brain for consistent learning targets
- Double DQN: Uses two networks to prevent overestimating action values
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from scripts.dqn import DQN
from scripts.replaybuffer import ReplayBuffer, replaybuffer_from_dict

STATE_DIM = 33   # 15 wall rays + 15 bomb rays + velocity + sin(angle) + cos(angle)
ACTION_DIM = 9   # 0=nothing, 1=forward, 2=back, 3=left, 4=right, 5-8=combinations

class DQNAgent:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Two networks: policy (learning) and target (stable reference)
        self.policy_net = DQN(STATE_DIM, ACTION_DIM, device=self.device).to(self.device)
        self.target_net = DQN(STATE_DIM, ACTION_DIM, device=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target never trains directly

        # Hyperparameters (tuning knobs)
        self.gamma = 0.95           # How much to value future rewards (0-1)
        self.lr = 0.0001            # Learning rate (how big steps to take)
        self.batch_size = 64        # Experiences to learn from at once
        
        # Exploration schedule
        self.epsilon = 1.0          # Start with 100% random
        self.epsilon_min = 0.1      # Never go below 10% random
        self.epsilon_decay = 0.9995 # Multiply epsilon by this each step
        
        self.target_update = 200    # Sync target network every N steps
        self.episode_count = 0

        # Memory of past experiences
        self.replay_buffer = ReplayBuffer(capacity=50000)
        
        # Optimizer and learning rate scheduler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=500, min_lr=1e-6
        )

        # Tracking
        self.train_step = 0
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "model.pt")
        
        # Performance records
        self.best_reward = -float('inf')
        self.best_finish_time = 0.0
        self.best_finish_episode = 0
        self.recent_rewards = []
        self.recent_checkpoints = []
        self.recent_finish_times = []

    def get_action(self, state, training=True):
        if state is None:
            return 0
        
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        else:
            with torch.no_grad():
                # Create directly on device instead of .to()
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def update(self):
        """Learn from a batch of past experiences"""
        if len(self.replay_buffer) < self.batch_size * 2:
            return None  # Not enough experiences yet

        # Sample random batch from memory
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        rewards = torch.clamp(rewards, -20.0, 20.0)  # Prevent extreme values

        # Current Q-values for actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use policy net to pick best action, target net to evaluate it
        with torch.no_grad():
            best_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate loss and update
        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Periodically sync target network
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def end_episode(self, episode_reward=0, checkpoints_reached=0, time_remaining=0, finished=False):
        """Called when episode ends - track stats and update learning rate"""
        self.episode_count += 1
        
        self.recent_rewards.append(episode_reward)
        self.recent_checkpoints.append(checkpoints_reached)
        
        # Keep only last 100
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
            self.recent_checkpoints.pop(0)
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        # Track finish times
        if finished:
            self.recent_finish_times.append(time_remaining)
            if len(self.recent_finish_times) > 100:
                self.recent_finish_times.pop(0)
            
            if time_remaining > self.best_finish_time:
                old_best = self.best_finish_time
                self.best_finish_time = time_remaining
                self.best_finish_episode = self.episode_count
                print(f"\n{'='*60}")
                print(f"🏆 NEW BEST! Time: {25.0 - time_remaining:.2f}s (was {25.0 - old_best:.2f}s)")
                print(f"{'='*60}\n")
        
        # Adjust learning rate if plateauing
        if len(self.recent_rewards) >= 100:
            self.scheduler.step(np.mean(self.recent_rewards))
        
        return self.epsilon

    def save_model(self, save_path=None):
        """Save everything needed to resume training"""
        if save_path is None:
            save_path = self.model_path
            
        checkpoint = {
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_count': self.episode_count,
            'best_reward': self.best_reward,
            'best_finish_time': self.best_finish_time,
            'best_finish_episode': self.best_finish_episode,
            'recent_rewards': self.recent_rewards,
            'recent_checkpoints': self.recent_checkpoints,
            'recent_finish_times': self.recent_finish_times,
            'replay_buffer': self.replay_buffer.to_dict(),
        }
        
        # Safe save with temp file (prevents corruption)
        tmp = save_path + '.tmp'
        torch.save(checkpoint, tmp)
        
        import time
        for attempt in range(3):
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(tmp, save_path)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    print("Warning: Could not save (file locked)")
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except:
                            pass

    def load_model(self, filepath=None):
        """Load saved training state"""
        if filepath is None:
            filepath = self.model_path
        if not os.path.exists(filepath):
            print("No model file found.")
            return False

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint.get('target_state_dict', checkpoint['model_state_dict']))

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                pass
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step = checkpoint.get('train_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.best_reward = checkpoint.get('best_reward', -float('inf'))
        self.best_finish_time = checkpoint.get('best_finish_time', 0.0)
        self.best_finish_episode = checkpoint.get('best_finish_episode', 0)
        self.recent_rewards = checkpoint.get('recent_rewards', [])
        self.recent_checkpoints = checkpoint.get('recent_checkpoints', [])
        self.recent_finish_times = checkpoint.get('recent_finish_times', [])

        if 'replay_buffer' in checkpoint and checkpoint['replay_buffer']:
            self.replay_buffer = replaybuffer_from_dict(checkpoint['replay_buffer'])

        print(f"Loaded: ep={self.episode_count}, ε={self.epsilon:.3f}, best={self.best_finish_time:.1f}s")
        return True