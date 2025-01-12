
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
        self.batch_size = 10

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

        # print("Actions", actions)
        for agent, action in actions.items():
            action_name = "COOPERATE" if action == COOPERATE else "DEFECT"
            print(f"{agent}: {action_name}")

        return actions


    def store_transition(self, state, action, reward, next_state, done, agent_name):
        # Übergang speichern
        transition = (state, action, reward, next_state, done)
        #print("Transisiton: ", transition, agent_name)
        
        self.memory_dict[agent_name].append(transition)



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
        # Sample a batch of transitions from memory
        batch_dict = self.sample_batch()

        if batch_dict is None:
            return  # Skip training if there aren't enough samples

        # Prepare batches for each agent
        for agent_name, batches in batch_dict.items():
            states, actions, rewards, next_states, dones = batches

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # Filter out transitions where dones (truncation flags) are 1
            non_truncated_indices = torch.where(dones.squeeze() == 0)[0]
            if len(non_truncated_indices) == 0:
                continue  # Skip training if no valid transitions remain

            states = states[non_truncated_indices]
            actions = actions[non_truncated_indices]
            rewards = rewards[non_truncated_indices]
            next_states = next_states[non_truncated_indices]
            dones = dones[non_truncated_indices]

            batch_dict[agent_name] = (states, actions, rewards, next_states, dones)

        # Train in batches
        for i in range(math.ceil(len(states) / self.batch_size)):
            for agent in self.agents:
                #print("Agent:", agent)

                # Extract mini-batch for current agent
                batch_states = tuple([
                    batch_dict[j][0][i * self.batch_size:(i + 1) * self.batch_size]
                    for j in self.agents
                ])
                q_values = self.model.forward(*batch_states)
                #print("q_vals1:", q_values)

                # Split q_values for each agent
                q_values = torch.tensor_split(q_values, [2], dim=1)
                #print("q_vals2:", q_values)

                # Select q_values corresponding to the actions taken
                q_values_selected = q_values[self.q_value_indices[agent]].gather(
                    1,
                    batch_dict[agent][1][i * self.batch_size:(i + 1) * self.batch_size]
                ).float()

                #print("Selected q_values:", q_values_selected)

                # Calculate target values using rewards
                rewards_batch = batch_dict[agent][2][i * self.batch_size:(i + 1) * self.batch_size].float()

                # Calculate loss
                loss = torch.nn.functional.mse_loss(q_values_selected, rewards_batch)
                #print(f"Loss for {agent}: {loss.item()}")

                # Backpropagation
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

        # Apply epsilon decay after training iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
