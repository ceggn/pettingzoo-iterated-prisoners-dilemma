import functools
import random
import gymnasium
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

COOPERATE = 0
DEFECT = 1

MOVES = ["COOPERATE", "DEFECT"]
NUM_ITERS = 25
REWARD_MAP = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 4),
    (DEFECT, COOPERATE): (4, 0),
    (DEFECT, DEFECT): (1, 1),
}

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = parallel_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    return env

def raw_env(render_mode=None):
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "ipd_v1"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(3, start=-1)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

    def render(self):
        if self.render_mode is None:
            return
        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]][0]], MOVES[self.state[self.agents[1]][0]]
            )
        else:
            string = "Game over"
        print(string)

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: [random.randint(0, 1)] for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Calculate rewards
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = REWARD_MAP[
            (actions[self.agents[0]], actions[self.agents[1]])
        ]

        # Check truncations and terminations
        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}
        terminations = {agent: False for agent in self.agents}

        if env_truncation:
            print("Game Over!")
            self.agents = []

        # Update observations
        observations = {
            self.agents[i]: [int(actions[self.agents[1 - i]])]
            for i in range(len(self.agents))
        }
        self.state = observations
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos