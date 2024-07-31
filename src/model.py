import torch.nn as nn
import numpy as np
import torch

class QLearning(nn.Module):
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        super(QLearning, self).__init__()
        # Initialize the parameters for Q-Learning
        self.n_states = n_states  # Number of states
        self.n_actions = n_actions  # Number of actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Define the neural network model
        self.model = nn.Sequential(
            nn.Linear(self.n_states, 64),  # First hidden layer with 64 units
            nn.ReLU(),  # Activation function
            nn.Linear(64, 128),  # Second hidden layer with 128 units
            nn.ReLU(),  # Activation function
            nn.Linear(128, n_actions)  # Output layer with number of actions
        )

        # Define the optimizer for the neural network
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=alpha)
        # Define the loss criterion
        self.criterion = nn.MSELoss()

    def forward(self, observation):
        # Forward pass through the network
        out = self.model(observation)
        return out