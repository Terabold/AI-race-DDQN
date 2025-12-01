# the neural network - המוח של הAI
# input: 33 numbers (rays + speed + direction)
# output: 9 action scores
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, device=None):
        super(DQN, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # simple 3-layer network: 33 -> 128 -> 128 -> 9
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        # numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        else:
            x = x.to(self.device)
        
        # leaky relu keeps small gradient for negative values
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)  # raw q-values, no activation
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))