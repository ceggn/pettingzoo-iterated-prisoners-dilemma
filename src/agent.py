
import torch
from collections import deque
import numpy as np
from model import QLearning
from vqc import VQC
import math

COOPERATE = 0
DEFECT = 1

class Agent:
    def __init__(self, observation_length, n_actions, n_games=25, alpha=0.1, epsilon=0.05, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01, model_type="vqc") -> None:
        # Initialize agent parameters
        self.n_games = n_games  # Number of games 
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum value for epsilon
        self.actions = n_actions  # Number of possible actions
        self.model_type = model_type

        if self.model_type == "q_learning":
            # Initialize Q-learning model
            self.model = QLearning(observation_length, n_actions, alpha, gamma, epsilon)
        elif self.model_type == "vqc":
            # Initialize the VQC
            self.model = VQC(observation_length=observation_length, num_layers=4, action_space=n_actions)

        # Increase memory size for experience replay
        self.memory = deque(maxlen=1000)  # Increased memory size
        
        # Define batch size for training
        self.batch_size = 64

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            action = np.random.randint(self.actions)
        else:
            # Exploit: choose the action with the highest Q-value
            state = torch.tensor([state], dtype=torch.float32)
            q_values = self.model.forward(state)
            action = torch.argmax(q_values).item()
        return action

    def update(self, state, action, reward, next_state):
        # Update the Q-table using the Bellman equation
        best_next_action = np.argmax(self.model.q_table[next_state])
        td_target = reward + self.gamma * self.model.q_table[next_state, best_next_action]
        td_error = td_target - self.model.q_table[state, action]
        self.model.q_table[state, action] += self.alpha * td_error

    def store_transition(self, state, action, reward, next_state, done):
        # Store the transition in memory
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        # Sample a batch of transitions from memory
        if len(self.memory) < self.batch_size:
            return None
        batch_indices = np.arange(0, len(self.memory))
        np.random.shuffle(batch_indices)
        batch = [self.memory[idx] for idx in batch_indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        # Ensure arrays are converted properly
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones
    
    def set_epsilon(self, val):
        self.epsilon = val

    
    
    def train(self):
        if self.model_type == "q_learning":
            # Train the Q-learning model using sampled batch
            batches = self.sample_batch()
            if batches is None:
                return

            states, actions, rewards, next_states, dones = batches
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            for i in range(math.ceil(len(states)/self.batch_size)):
                current_q_values = self.model.forward(states[i*self.batch_size:(i+1)*self.batch_size])
                # Update logic can be placed here for q_learning

        elif self.model_type == "vqc":
            # Train the VQC model using the optimizer
            batches = self.sample_batch()
            if batches is None:
                return

            states, actions, rewards, next_states, dones = batches
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            for i in range(math.ceil(len(states)/self.batch_size)):
                batch_states = states[i*self.batch_size:(i+1)*self.batch_size]
                q_values = self.model.forward(batch_states)



                # Ensure q_values and rewards have the same shape
                # Select the q_values corresponding to the actions taken
                q_values = q_values.gather(1, actions[i*self.batch_size:(i+1)*self.batch_size])

                # Convert q_values and rewards to the same dtype (Float) for consistency
                q_values = q_values.float()
                rewards_batch = rewards[i*self.batch_size:(i+1)*self.batch_size].float()

                # Calculate loss and perform optimization step
                loss = torch.nn.functional.mse_loss(q_values, rewards_batch)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                
        # Apply epsilon decay after each training iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


