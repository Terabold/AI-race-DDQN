import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from scripts.dqn import DQN
from scripts.replaybuffer import ReplayBuffer, replaybuffer_from_dict

STATE_DIM = 33 
ACTION_DIM = 9

class DQNAgent:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQN(STATE_DIM, ACTION_DIM, device=self.device).to(self.device)
        self.target_net = DQN(STATE_DIM, ACTION_DIM, device=self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # SIMPLIFIED HYPERPARAMETERS
        self.gamma = 0.95  # Slightly lower discount
        self.lr = 0.0001  # Slightly higher learning rate
        self.batch_size = 64  # Smaller batches = more updates
        
        # EPSILON SCHEDULE - Decay faster to stable exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.9995  # Faster decay to min
        
        # Target network update
        self.target_update = 200  # Update target less often
        
        # Episode tracking
        self.episode_count = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=50000)  # Smaller buffer
        
        # Optimizer - Standard Adam (not AdamW)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Learning rate scheduler - More aggressive
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,  # Cut LR in half when plateauing
            patience=500,  # Reduce sooner
            min_lr=1e-6,
            verbose=True
        )

        # Tracking
        self.train_step = 0
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "model.pt")
        
        # Performance tracking
        self.best_reward = -float('inf')
        self.best_finish_time = 0.0
        self.best_finish_episode = 0
        self.episodes_without_improvement = 0
        self.recent_rewards = []
        self.recent_checkpoints = []
        self.recent_finish_times = []

    def get_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy
        """
        if state is None:
            return 0
        
        if training and random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def update(self):
        """
        Update the policy network using a batch from replay buffer
        """
        if len(self.replay_buffer) < self.batch_size * 2:  # Start training sooner
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Clip rewards
        rewards = torch.clamp(rewards, -20.0, 20.0)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Huber loss
        loss = F.smooth_l1_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # More aggressive clipping
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def end_episode(self, episode_reward=0, checkpoints_reached=0, time_remaining=0, finished=False):
        """
        Called when an episode ends
        """
        self.episode_count += 1
        
        # Track recent performance
        self.recent_rewards.append(episode_reward)
        self.recent_checkpoints.append(checkpoints_reached)
        
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
                
                improvement = time_remaining - old_best
                print(f"\n{'='*80}")
                print(f"🏆 NEW BEST FINISH TIME! 🏆")
                print(f"  Episode: {self.episode_count}")
                print(f"  Time Remaining: {time_remaining:.2f}s / 25.0s")
                print(f"  Completion Time: {25.0 - time_remaining:.2f}s (Previous best: {25.0 - old_best:.2f}s)")
                print(f"  Improvement: {improvement:.2f}s faster")
                print(f"{'='*80}\n")
        
        # Update learning rate
        if len(self.recent_rewards) >= 100:
            avg_reward = np.mean(self.recent_rewards)
            self.scheduler.step(avg_reward)
        
        return self.epsilon

    def save_model(self, save_path=None):
        """
        Save model, optimizer, and replay buffer
        """
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
                    print(f"Warning: Could not save model (file locked). Will retry next save.")
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except:
                            pass

    def load_model(self, filepath=None):
        """
        Load model and training state
        """
        if filepath is None:
            filepath = self.model_path
        if not os.path.exists(filepath):
            print("No model file found.")
            return False

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        
        if 'target_state_dict' in checkpoint:
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("Warning: Could not load scheduler state")
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']

        if 'train_step' in checkpoint:
            self.train_step = checkpoint['train_step']
            
        if 'episode_count' in checkpoint:
            self.episode_count = checkpoint['episode_count']
        
        if 'best_reward' in checkpoint:
            self.best_reward = checkpoint['best_reward']
        
        if 'best_finish_time' in checkpoint:
            self.best_finish_time = checkpoint['best_finish_time']
        
        if 'best_finish_episode' in checkpoint:
            self.best_finish_episode = checkpoint['best_finish_episode']
        
        if 'recent_rewards' in checkpoint:
            self.recent_rewards = checkpoint['recent_rewards']
            
        if 'recent_checkpoints' in checkpoint:
            self.recent_checkpoints = checkpoint['recent_checkpoints']
        
        if 'recent_finish_times' in checkpoint:
            self.recent_finish_times = checkpoint['recent_finish_times']
        else:
            self.recent_finish_times = []

        if 'replay_buffer' in checkpoint and checkpoint['replay_buffer']:
            self.replay_buffer = replaybuffer_from_dict(checkpoint['replay_buffer'])

        current_lr = self.optimizer.param_groups[0]['lr']

        print(f"Loaded: ε={self.epsilon:.4f}, step={self.train_step}, ep={self.episode_count}, buffer={len(self.replay_buffer)}")
        print(f"Current LR: {current_lr:.6f}")
        print(f"Best reward: {self.best_reward:.1f}")
        print(f"Best finish time: {self.best_finish_time:.2f}s remaining (Episode {self.best_finish_episode})")
        
        if self.recent_rewards:
            print(f"Recent avg reward: {np.mean(self.recent_rewards):.1f}")
        if self.recent_checkpoints:
            print(f"Recent avg checkpoints: {np.mean(self.recent_checkpoints):.1f}/25")
        if self.recent_finish_times:
            print(f"Recent avg finish time: {np.mean(self.recent_finish_times):.2f}s remaining")
        
        return True