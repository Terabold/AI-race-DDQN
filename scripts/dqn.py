import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize the DQN model
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Size of hidden layer (256 like in DDQN Keras)
            device (torch.device): Device to run the model on (CPU/CUDA)
        """
        super(DQN, self).__init__()
        
        # Store device
        self.device = device
        
        # Using a simpler network structure like in DDQN Keras
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.MSELoss = nn.MSELoss()
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        else:
            x = x.to(self.device)
            
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        q_values = self.fc3(x)
        
        return q_values
    
    def save(self, filepath):
        """
        Save the model weights to a file
        
        Args:
            filepath (str): Path to save the model
        """
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath):
        """
        Load model weights from a file
        
        Args:
            filepath (str): Path to load the model from
        """
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        
    def __call__(self, x):
        """
        Allow direct calling of model with inputs
        
        Args:
            x: Input state
            
        Returns:
            torch.Tensor: Q-values
        """
        return self.forward(x).to(self.device)