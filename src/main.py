import os
import ipd_env
from agent import Agent
import numpy as np
import torch as T
import json
import argparse

NUM_GAMES_PER_SEGMENT = 25

# Function to run a single game and return rewards and actions
def run_single_game(seed):
    np.random.seed(seed)
    T.manual_seed(seed)

    # Initialize the environment
    env = ipd_env.parallel_env(render_mode="human")

    # Define state and action spaces
    observation_length = 1
    n_actions = 2

    # Initialize agents for each possible agent in the environment
    agents = {agent_id: Agent(observation_length, n_actions) for agent_id in env.possible_agents}

    rewards_p1 = []
    rewards_p2 = []
    actions = {agent_id: [] for agent_id in env.possible_agents}

    # Game loop for 100 steps
    n = 10
    while n > 0:
        n -= 1
        observations, infos = env.reset()
        while env.agents:
            actions_step = {agent_id: agents[agent_id].choose_action(observations[agent_id]) for agent_id in env.agents}
            new_observations, rewards, terminations, truncations, infos = env.step(actions_step)

            if len(env.agents) >= 2:
                rewards_p1.append(rewards[env.agents[0]])
                rewards_p2.append(rewards[env.agents[1]])

            for agent_id in env.agents:
                actions[agent_id].append(actions_step[agent_id])
                # Store transitions in memory for each agent
                agents[agent_id].store_transition(
                observations[agent_id],  # Current state
                actions_step[agent_id],  # Action taken
                rewards[agent_id],       # Reward received
                new_observations[agent_id],  # New state after action
                terminations[agent_id]   # Whether the episode has ended
            )

            observations = new_observations

        for i in agents.values(): i.train()

        print("Game: ", n)

    env.close()

    return rewards_p1, rewards_p2, actions


# Function to run multiple games and calculate average rewards per 25-game segment
def run_multiple_games(num_runs, output_dir, seed):
    all_runs_rewards = {}

    for i in range(num_runs):
        seed = i + 1
        rewards_p1, rewards_p2, _ = run_single_game(seed)

        # Group the rewards into segments of 25 games
        avg_rewards_p1 = []
        avg_rewards_p2 = []

        for start in range(0, len(rewards_p1), NUM_GAMES_PER_SEGMENT):
            end = start + NUM_GAMES_PER_SEGMENT
            segment_p1 = rewards_p1[start:end]
            segment_p2 = rewards_p2[start:end]

            # Calculate the average reward for each segment
            avg_reward_p1 = np.mean(segment_p1) if segment_p1 else 0
            avg_reward_p2 = np.mean(segment_p2) if segment_p2 else 0

            avg_rewards_p1.append(avg_reward_p1)
            avg_rewards_p2.append(avg_reward_p2)

        # Store the average rewards for this run
        all_runs_rewards[f"Run_{i + 1}"] = {
            "Player_1_Average_Rewards": avg_rewards_p1,
            "Player_2_Average_Rewards": avg_rewards_p2
        }

    # Save the detailed rewards data to a JSON file
    with open(os.path.join(output_dir, "average_rewards_per_25_game_segment.json"), 'w') as f:
        json.dump(all_runs_rewards, f, indent=4)

    return all_runs_rewards


# Main function to execute multiple runs and save rewards and actions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="idp")
    parser.add_argument("-s", type=int, help="Seeding")
    args = parser.parse_args()
    seed = args.s        
    num_runs = 2  # Number of runs to perform
    output_dir = "results_" + str(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run multiple games and save results
    run_multiple_games(1, output_dir, seed)