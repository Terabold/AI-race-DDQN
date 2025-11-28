"""
DQN.PY - The AI's "brain"
A neural network that learns to predict which action is best
Input: Car sensors (30 values) + speed + angle = 33 values
Output: Score for each of 9 possible actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, device=None):
        super(DQN, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Three layers: input -> hidden -> hidden -> output
        self.fc1 = nn.Linear(state_dim, hidden_dim)   # 33 -> 128
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 128 -> 128
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 128 -> 9
        
    def forward(self, x):
        """Run input through network, get action scores"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        else:
            x = x.to(self.device)
        
        # LeakyReLU activation (allows small negative values)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)  # Raw scores (Q-values)
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))