import numpy as np
import torch
import torch.nn as nn

class QLearning:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((n_states, n_actions))

        self.model = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        # Define the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=alpha)
        self.criterion = nn.HuberLoss()    

    def forward(self, observation):
        out = self.model(torch.tensor(observation, dtype=torch.float32))
        return out