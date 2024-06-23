import numpy as np
from gym.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
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
        return {agent: self.state for agent in self.agents}

    def step(self, action):
        if self.dones[self.agents[0]] or self.dones[self.agents[1]]:
            raise Exception("Step called after episode is done")

        ac0, ac1 = action["player_0"], action["player_1"]
        self.step_count += 1

        rewards = [self.payout_mat[ac1][ac0], self.payout_mat[ac0][ac1]]
        self.rewards["player_0"], self.rewards["player_1"] = rewards

        state = np.zeros(5)
        state[ac0 * 2 + ac1] = 1
        self.state = state

        if self.step_count >= self.max_steps:
            self.dones = {agent: True for agent in self.agents}

        return {agent: self.state for agent in self.agents}, self.rewards, self.dones, self.infos

    def render(self, mode='human'):
        print(f"Step: {self.step_count}")
        print(f"State: {self.state}")
        print(f"Rewards: {self.rewards}")

    def close(self):
        pass
