import torch
from collections import deque
import numpy as np
from model import QLearning  # Assuming this is the correct import path

COOPERATE = 0
DEFECT = 1

class Agent:
    def __init__(self, n_states, n_actions, n_games=10, alpha=0.003, epsilon=0.05, gamma=0.99, actions=2) -> None:
        self.n_games = n_games  # number of games
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # randomness
        self.gamma = gamma  # discount rate
        self.actions = actions

        self.q_learning = QLearning(n_states, n_actions, alpha, gamma, epsilon)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.actions)  # Explore
        else:
            return np.argmax(self.q_learning.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_learning.q_table[next_state])
        td_target = reward + self.gamma * self.q_learning.q_table[next_state, best_next_action]
        td_error = td_target - self.q_learning.q_table[state, action]
        self.q_learning.q_table[state, action] += self.alpha * td_error

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        if len(self.memory) < self.batch_size:
            return None
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def train(self, state, action, reward, next_state):
        batch = self.sample_batch()
        if batch is None:
            return

        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Get the current Q-values for all actions in current states
        current_q_values = self.q_learning.forward(states)

        # Select the Q-values corresponding to the actions taken
        chosen_q_values = current_q_values.gather(1, actions)

        # Get the Q-values for the next states
        next_q_values = self.q_learning.forward(next_states)

        # Calculate the target Q-values using the Bellman equation
        max_next_q_values, _ = next_q_values.max(dim=1, keepdim=True)
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute the loss
        loss = torch.nn.functional.mse_loss(chosen_q_values, target_q_values)

        # Perform a gradient descent step
        self.q_learning.optimizer.zero_grad()
        loss.backward()
        self.q_learning.optimizer.step()