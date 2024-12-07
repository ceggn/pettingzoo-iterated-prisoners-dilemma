
import torch
from collections import deque
import numpy as np
from model import QLearning
from vqc_combined import VQC_Combined
import math

COOPERATE = 0
DEFECT = 1

class Combined_Agents:
    def __init__(self, agents, observation_length, n_actions, n_games=25, alpha=0.1, epsilon=0.05, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01) -> None:
        # Initialize agent parameters
        self.n_games = n_games  # Number of games 
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum value for epsilon
        self.actions = n_actions  # Number of possible actions
        self.agents = agents # Names of agents as list of strings

        self.q_value_indices = {agent: c for c, agent in enumerate(self.agents)}

        # Initialize the VQC
        self.model = VQC_Combined(observation_length=observation_length, num_layers=4, action_space=n_actions)

        # Increase memory size for experience replay
        self.memory_dict = {name : deque(maxlen=1000) for name in self.agents}  # Increased memory size for n agents

        # Define batch size for training
        self.batch_size = 64

    def choose_actions(self, state_dict):
        # Epsilon-greedy action selection

        action_required = []
        actions = {}

        for agent in self.agents:

            if np.random.rand() < self.epsilon:
                # Explore: choose a random action
                actions[agent] = np.random.randint(self.actions)
            else:
                action_required.append(agent)
  
        
        if len(action_required)>0: 
            # Exploit: choose the action with the highest Q-value
            states = tuple([torch.tensor([state_dict[i]], dtype=torch.float32) for i in self.agents])
            q_values = self.model.forward(*states)

            q_values = torch.tensor_split(q_values, [2], dim=1)

            for agent in action_required:
                actions[agent] = torch.argmax(q_values[self.q_value_indices[agent]]).item()

        return actions


    def store_transition(self, state, action, reward, next_state, done, agent_name):
        # Store the transition in memory
        self.memory_dict[agent_name].append((state, action, reward, next_state, done))


    def sample_batch(self):
        # Sample a batch of transitions from memory

        for memory in self.memory_dict.values():
            if len(memory) < self.batch_size:
                return None
        batch_indices = np.arange(0, len(next(iter(self.memory_dict.values()))))
        np.random.shuffle(batch_indices)

        batch_dict = {}

        for agent in self.agents:
            batch = [self.memory_dict[agent][idx] for idx in batch_indices]
            states, actions, rewards, next_states, dones = zip(*batch)
            # Ensure arrays are converted properly
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)

            batch_dict[agent]= (states, actions, rewards, next_states, dones)

        return batch_dict
    
    def set_epsilon(self, val):
        self.epsilon = val

    def train(self):

        # Train the VQC model using the optimizer
        batch_dict = self.sample_batch()
       
        if batch_dict is None:
            return


        for agent_name, batches in batch_dict.items():

            states, actions, rewards, next_states, dones = batches
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            batch_dict[agent_name] = (states, actions, rewards, next_states, dones)


        for agent in self.agents:

            for i in range(math.ceil(len(states)/self.batch_size)):
                batch_states = tuple([batch_dict[j][0][i*self.batch_size:(i+1)*self.batch_size] for j in self.agents])
                q_values = self.model.forward(*batch_states)

                q_values = torch.tensor_split(q_values, [2], dim=1)
                # Ensure q_values and rewards have the same shape
                # Select the q_values corresponding to the actions taken


                q_values = q_values[self.q_value_indices[agent]].gather(1, batch_dict[agent][1][i*self.batch_size:(i+1)*self.batch_size]).float()
                rewards_batch = rewards[i*self.batch_size:(i+1)*self.batch_size].float()

                # Calculate loss and perform optimization step
                loss = torch.nn.functional.mse_loss(q_values, rewards_batch)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                
        # Apply epsilon decay after each training iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


