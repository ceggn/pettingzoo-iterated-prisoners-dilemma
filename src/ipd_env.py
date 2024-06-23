import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils.env import ParallelEnv


class OneHot(Box):
    def __init__(self, n):
        super().__init__(0, 1, (n,), dtype=np.float32)
        self.n = n

    def sample(self):
        one_hot = np.zeros(self.n, dtype=np.float32)
        one_hot[np.random.randint(self.n)] = 1
        return one_hot

    def contains(self, x):
        return isinstance(x, np.ndarray) and x.shape == (self.n,) and np.all((x == 0) | (x == 1))

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class IteratedPrisonersDilemma(AECEnv):
    """
    A two-agent environment for the Prisoner's Dilemma game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps):
        super().__init__()
        self.max_steps = max_steps
        self.payout_mat = np.array([[-1., 0.], [-3., -2.]])
        self.agents = ["player_0", "player_1"]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        self.action_spaces = {agent: Discrete(2) for agent in self.agents}
        self.observation_spaces = {agent: OneHot(5) for agent in self.agents}
        self.step_count = None
        self.last_actions = None  # To store the last actions taken

        # Q-learning parameters
        self.q_nets = {agent: QNetwork(5, 2) for agent in self.agents}
        self.optimizers = {agent: optim.Adam(self.q_nets[agent].parameters(), lr=0.01) for agent in self.agents}
        self.loss_fn = nn.MSELoss()
        self.discount_factor = 0.95
        self.epsilon = 0.1

    @property
    def num_agents(self):
        return len(self.agents)

    def reset(self):
        self.step_count = 0
        self.agents = ["player_0", "player_1"]
        init_state = np.zeros(5)
        init_state[-1] = 1
        self.state = init_state
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.last_actions = None
        return {agent: self.state for agent in self.agents}

    def step(self, action):
        if self.dones[self.agents[0]] or self.dones[self.agents[1]]:
            raise Exception("Step called after episode is done")

        ac0, ac1 = action["player_0"], action["player_1"]
        self.step_count += 1
        self.last_actions = (ac0, ac1)

        rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]
        self.rewards["player_0"], self.rewards["player_1"] = rewards

        state = np.zeros(5)
        state[ac0 * 2 + ac1] = 1
        self.state = state

        if self.step_count >= self.max_steps:
            self.dones = {agent: True for agent in self.agents}

        self.update_q_nets(ac0, ac1, rewards)

        return {agent: self.state for agent in self.agents}, self.rewards, self.dones, self.infos

    def update_q_nets(self, ac0, ac1, rewards):
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(self.state).unsqueeze(0)

        for i, agent in enumerate(self.agents):
            action = ac0 if agent == "player_0" else ac1
            reward = rewards[i]

            q_values = self.q_nets[agent](state_tensor)
            with torch.no_grad():
                next_q_values = self.q_nets[agent](next_state_tensor)
            
            target_q_value = reward + self.discount_factor * next_q_values.max().item()
            loss = self.loss_fn(q_values[0, action], torch.FloatTensor([target_q_value]).unsqueeze(0))
            
            self.optimizers[agent].zero_grad()
            loss.backward()
            self.optimizers[agent].step()

    def select_action(self, agent):
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return self.action_spaces[agent].sample()
        else:
            with torch.no_grad():
                q_values = self.q_nets[agent](state_tensor)
            return q_values.argmax().item()

    def render(self, mode='human'):
        actions_str = f"Actions: Player 0: {self.last_actions[0]}, Player 1: {self.last_actions[1]}"
        print(f"Step: {self.step_count}")
        print(f"State: {self.state}")
        print(actions_str)
        print(f"Rewards: {self.rewards}")

    def close(self):
        pass

def main():
    env = IteratedPrisonersDilemma(max_steps=10)
    obs = env.reset()
    
    for _ in range(10):
        actions = {
            "player_0": env.select_action("player_0"),
            "player_1": env.select_action("player_1")
        }
        obs, rewards, dones, infos = env.step(actions)
        env.render()
        if all(dones.values()):
            break

    env.close()

if __name__ == "__main__":
    main()
