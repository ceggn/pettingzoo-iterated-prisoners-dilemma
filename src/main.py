import os
import ipd_env
from agent import Agent
from agent_combined import Combined_Agents
import numpy as np
import torch as T
import json
import argparse
import random

#TODO: handle with Argparse
approach = "combined_vqc"

# Function to run a single game and return rewards and actions
def run_single_game(seed, output_dir):
    np.random.seed(seed)
    T.manual_seed(seed)
    random.seed(seed)

    # Initialize the environment
    env = ipd_env.parallel_env(render_mode="human")

    # Define state and action spaces
    observation_length = 1
    n_actions = 2

    # Initialize agents for each possible agent in the environment
    if approach == "combined_vqc":
        combined_agents = Combined_Agents(env.possible_agents, observation_length, n_actions)
    else:
        agents = {agent_id: Agent(observation_length, n_actions) for agent_id in env.possible_agents}

    rewards_p1 = []
    rewards_p2 = []
    actions = {agent_id: [] for agent_id in env.possible_agents}

    # Game loop for 100 games
    n = 200
    while n > 0:
        n -= 1
        observations, infos = env.reset(seed=seed)
        while env.agents:
            if approach == "combined_vqc":
                actions_step = combined_agents.choose_actions(observations)
            else: 
                actions_step = {agent_id: agents[agent_id].choose_action(observations[agent_id]) for agent_id in env.agents}
            
            new_observations, rewards, terminations, truncations, infos = env.step(actions_step)

            if len(env.agents) >= 2:
                rewards_p1.append(rewards[env.agents[0]])
                rewards_p2.append(rewards[env.agents[1]])

            for agent_id in env.agents:
                actions[agent_id].append(actions_step[agent_id])
                # Store transitions in memory for each agent
                if approach =="combined_vqc":
                    combined_agents.store_transition(
                    observations[agent_id],
                    actions_step[agent_id],
                    rewards[agent_id],
                    new_observations[agent_id],
                    terminations[agent_id],
                    agent_id
                    )

                else:    
                    agents[agent_id].store_transition(
                        observations[agent_id],
                        actions_step[agent_id],
                        rewards[agent_id],
                        new_observations[agent_id],
                        terminations[agent_id]
                    )

            observations = new_observations


        if approach == "combined_vqc":

            combined_agents.train()
        else: 
            for agent in agents.values():
                agent.train()

    env.close()

    # Store the full rewards for this run
    all_runs_rewards = {
        "Player_1_Rewards": rewards_p1,
        "Player_2_Rewards": rewards_p2
    }

    # Save the complete rewards data to a JSON file
    with open(os.path.join(output_dir, "complete_rewards.json"), 'w') as f:
        json.dump(all_runs_rewards, f, indent=4)
    
    return rewards_p1, rewards_p2, actions



# Main function to execute multiple runs and save rewards and actions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="idp")
    parser.add_argument("-s", type=int, help="Seeding", default=42)
    args = parser.parse_args()
    seed = args.s
    
    if approach == "combined_vqc":
        output_dir = "experiment_cnot_run10_" + str(seed)
    else:
        output_dir = "results_" + str(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    run_single_game(seed, output_dir)