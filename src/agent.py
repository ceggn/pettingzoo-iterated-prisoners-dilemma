import math
import random
from collections import deque

import numpy as np
import torch as T

from cmodel import cmodel
from qmodel import qmodel

# Set default torch tensor type to float64 to circumvent issue with amplitude encoding normalization
T.set_default_dtype(T.float64)

class Agent:
    def __init__(self, buffer_len, model="classical", obs_window_length=5, n_actions=2, learning_rate=0.003, epsilon=0.05, n_batches=5) -> None:
        self.n_actions = n_actions
        self.obs_window_length = obs_window_length
        self.n_batches = n_batches
        self.buffer_len = buffer_len
        
        self.gamma = 0.99
        self.epsilon = epsilon

        if model == "classical":
            self.model = cmodel(input_size=obs_window_length * 2, output_size=n_actions, learning_rate=learning_rate)
        elif model == "quantum":
            self.model = qmodel(input_size=obs_window_length * 2, output_size=n_actions, learning_rate=learning_rate)
        else:
            self.model = cmodel(input_size=obs_window_length * 2, output_size=n_actions, learning_rate=learning_rate)
            print("Unknown model provided. Selected classical model")

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.memory = deque(maxlen=buffer_len)

    def save_model(self, checkpoint_dir, name):
        self.model.save_checkpoint(checkpoint_dir, filename=name)

    def load_model(self, checkpoint_dir, name):
        self.model.load_checkpoint(checkpoint_dir, filename=name)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def preprocess_observation(self, observation):
        return np.array(observation).flatten()

    def choose_action(self, observation):
        observation = self.preprocess_observation(observation)
        observation_tensor = T.tensor([observation], dtype=T.float64).to(self.device)
        
        # Get action
        Q_values = self.model.forward(observation_tensor)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return T.argmax(Q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.n_batches:
            return

        batch = random.sample(self.memory, self.n_batches)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = T.tensor(states, dtype=T.float64).to(self.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float64).to(self.device)
        next_states = T.tensor(next_states, dtype=T.float64).to(self.device)
        dones = T.tensor(dones, dtype=T.float64).to(self.device)

        current_Q_values = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_Q_values = self.model.forward(next_states).max(1)[0]
        target_Q_values = rewards + (self.gamma * next_Q_values * (1 - dones))

        loss = self.model.criterion(current_Q_values, target_Q_values)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()