import torch
from collections import deque
import numpy as np
from model import QLearning
from vqc import VQC
import math

COOPERATE = 0
DEFECT = 1

class Agent:
    def __init__(self, observation_length, n_actions, n_games=25, alpha=0.1, epsilon=0.05, gamma=0.99, epsilon_decay=0.99, epsilon_min=0.01, model_type ="vqc") -> None:
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
        if self.model_type == "vqc":
            # Initialize the VQC
            self.model = VQC(num_qubits=observation_length, num_layers=2, action_space=n_actions)

        
        # Initialize memory for experience replay
        self.memory = deque(maxlen=25)
        
        # Define batch size for training
        self.batch_size = 25

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            action = np.random.randint(self.actions)
        else:
            # Exploit: choose the action with the highest Q-value
            state = torch.tensor([state],dtype=torch.float32)
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
        # batch_size elemente , loop
        # batch[i:]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Erinerungen shufflen in batches unterteile und alle returnen
        # training for schleife for in batches 
        
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

            # Debugging: Print the shape and contents of the states tensor
            # print(f"Shape of states tensor: {states.shape}")
            # print(f"Contents of states tensor: {states}")

            # Get the current Q-values for all actions in current states
            current_q_values = self.model.forward(states)

            # Select the Q-values corresponding to the actions taken
            chosen_q_values = current_q_values.gather(1, actions)

            # Get the Q-values for the next states
            next_q_values = self.model.forward(next_states)

            # Calculate the target Q-values using the Bellman equation
            max_next_q_values, _ = next_q_values.max(dim=1, keepdim=True)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

            # Compute the loss
            loss = torch.nn.functional.mse_loss(chosen_q_values, target_q_values)

            # Perform a gradient descent step
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()