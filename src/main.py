import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural network for each agent
class DilemmaNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DilemmaNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

class Agent:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, initial_epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.model = DilemmaNet(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.action_to_idx = {'C': 0, 'D': 1}
        self.idx_to_action = {0: 'C', 1: 'D'}
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def sample_policy(self, state):
        if random.random() < self.epsilon:
            action_idx = random.choice([0, 1])
            output = torch.tensor([[0.5, 0.5]], dtype=torch.float32, requires_grad=True)
        else:
            output = self.model(state)
            action_idx = torch.argmax(output).item()
        return self.idx_to_action[action_idx], output
    
    def update_policy(self, output, action, reward):
        action_idx = self.action_to_idx[action]
        loss = self.criterion(output, torch.tensor([action_idx], dtype=torch.long))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def main():
    # Initialize agents
    input_size = 8  # History size + opponent's last move + own last move
    hidden_size = 128
    output_size = 2  # Cooperate or Defect

    agent1 = Agent(input_size, hidden_size, output_size)
    agent2 = Agent(input_size, hidden_size, output_size)

    # Define the reward matrix for the Prisoner's Dilemma
    reward_matrix = {
        ('C', 'C'): (3, 3),
        ('C', 'D'): (0, 5),
        ('D', 'C'): (5, 0),
        ('D', 'D'): (1, 1)
    }

    # List to store summed total rewards
    summed_rewards_per_game = []

    # Training the agents
    num_epochs = 5000  # Increase the number of epochs
    history_length = 3  # Number of previous moves to consider
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        history = [(0, 0)] * history_length
        total_reward_agent1 = 0
        total_reward_agent2 = 0
        total_loss_agent1 = 0
        total_loss_agent2 = 0

        total_reward_per_game = 0  # Reset total reward per game

        for t in range(10):
            # Flatten the history list
            flattened_history1 = [item for sublist in history[-history_length:] for item in sublist] + [history[-1][1], history[-1][0]]
            flattened_history2 = [item for sublist in history[-history_length:] for item in sublist] + [history[-1][0], history[-1][1]]

            # Create input for each agent
            input_agent1 = torch.tensor([flattened_history1], dtype=torch.float32)
            input_agent2 = torch.tensor([flattened_history2], dtype=torch.float32)

            # Agents' actions
            action1, output1 = agent1.sample_policy(input_agent1)
            action2, output2 = agent2.sample_policy(input_agent2)

            # Rewards
            reward1, reward2 = reward_matrix[(action1, action2)]

            # Update total reward per game
            total_reward_per_game += (reward1 + reward2)


            # Print actions taken and rewards received
            if epoch % 10 == 0 and t < 10:  # Print first 10 rounds for every 10th epoch
                print(f"Round {t}: Agent 1 -> {action1} ({reward1}), Agent 2 -> {action2} ({reward2})")


            # Update history
            history.append((agent1.action_to_idx[action1], agent2.action_to_idx[action2]))
            if len(history) > history_length:
                history.pop(0)

            # Update policies and track loss
            loss_agent1 = agent1.update_policy(output1, action1, reward1)
            loss_agent2 = agent2.update_policy(output2, action2, reward2)

            total_reward_agent1 += reward1
            total_reward_agent2 += reward2
            total_loss_agent1 += loss_agent1
            total_loss_agent2 += loss_agent2


        # Append the total reward per game for the current epoch
        summed_rewards_per_game.append(total_reward_per_game)

        # Decay epsilon 
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        if epoch % 100 == 0:
            print(f"Epsilon after {epoch} epochs: Agent 1: {agent1.epsilon}, Agent 2: {agent2.epsilon}")

        print(f"End of Epoch {epoch}: Total Reward Agent 1: {total_reward_agent1}, Total Reward Agent 2: {total_reward_agent2}")
        print(f"Total Loss Agent 1: {total_loss_agent1}, Total Loss Agent 2: {total_loss_agent2}")
        print("------")

    print("Training complete")

    # Plot the rewards per game over time
    plt.plot(summed_rewards_per_game, label='Reward Per Game (Agent 1 + Agent 2)')
    plt.xlabel('Epoch Number')
    plt.ylabel('Reward Per Game')
    plt.title('Reward Per Game Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
