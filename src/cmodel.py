import os

import torch as T
import torch.nn as nn
from torch.distributions import Categorical

# Set default torch tensor type to float64 to circumvent issue with amplitude encoding normalization
T.set_default_dtype(T.float64)

class cmodel(nn.Module):
    def __init__(self, input_size, output_size, learning_rate) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.fc_size = 1024
        
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, self.fc_size),
            nn.ReLU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.Linear(self.fc_size, self.output_size)
        )
        
        self.optimizer = T.optim.SGD(self.parameters(), lr=learning_rate)
        self.criterion = nn.HuberLoss()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        out = self.actor(observation)
        return out

    def save_checkpoint(self, checkpoint_path, filename="model") -> None:
        T.save(self.state_dict(), os.path.join(checkpoint_path, filename))

    def load_checkpoint(self, checkpoint_path, filename="model") -> None:
        self.load_state_dict(T.load(os.path.join(checkpoint_path, filename)))