import os
import ipd_env
from agent import Agent
import matplotlib.pyplot as plt
import json
import torch

# Initialize the environment
env = ipd_env.parallel_env(render_mode="human")

# Define state and action spaces
n_states = 2  # Number of possible states (example: 2 states)
n_actions = 2  # Number of possible actions (example: 2 actions)

# Initialize agents for each possible agent in the environment
agents = {agent_id: Agent(n_states, n_actions) for agent_id in env.possible_agents}

# Lists to store rewards
rewards_player_1 = []
rewards_player_2 = []
combined_rewards = []

# Dictionaries to store actions at each step
actions_log = {agent_id: [] for agent_id in env.possible_agents}

step = 0

n = 100
while n > 0:
    n -= 1
    observations, infos = env.reset()
    while env.agents:
        actions = {}
        steps = 0
        
        # Each agent chooses an action based on their current state
        for agent_id in env.agents:
            actions[agent_id] = agents[agent_id].choose_action(observations[agent_id])

        # Log actions as strings
        for agent_id in env.agents:
            action_str = "COOPERATE" if actions[agent_id] == 0 else "DEFECT"
            actions_log[agent_id].append(action_str)

        # Take a step in the environment using the chosen actions
        new_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store rewards for visualization, ensuring indices are valid
        if len(env.agents) >= 2:
            reward_1 = rewards[env.agents[0]]
            reward_2 = rewards[env.agents[1]]
            rewards_player_1.append(reward_1)
            rewards_player_2.append(reward_2)
            combined_rewards.append(reward_1 + reward_2)

        # Store the transitions in memory for each agent
        for agent_id in env.agents:
            agents[agent_id].store_transition(
                observations[agent_id],  # Current state
                actions[agent_id],       # Action taken
                rewards[agent_id],       # Reward received
                new_observations[agent_id],  # New state after action
                terminations[agent_id]   # Whether the episode has ended
            )

        # Print step info
        print(f"Step {step}:")
        for agent_id in env.agents:
            print(f"  Agent {agent_id} - Action: {actions[agent_id]}, Reward: {rewards[agent_id]}, Observation: {new_observations[agent_id]}")
            print(f"  Epsilon (exploration rate) for Agent {agent_id}: {agents[agent_id].epsilon}")
        print("")

        # Update observations for the next step
        observations = new_observations
        step += 1

    # Update Q-learning model every 25 steps
    for agent_id in agents.keys():
        agents[agent_id].train()
        agents[agent_id].epsilon = agents[agent_id].epsilon * agents[agent_id].epsilon_decay
        print(agents[agent_id].epsilon)

# Set epsilon to 0 for all agents
for agent_id in agents.keys():
    agents[agent_id].set_epsilon(0)

# Run additional episodes with epsilon set to 0
for i in range(10):
    observations, infos = env.reset()
    while env.agents:
        actions = {}
        steps = 0
        
        # Each agent chooses an action based on their current state
        for agent_id in env.agents:
            actions[agent_id] = agents[agent_id].choose_action(observations[agent_id])

        # Log actions as strings
        for agent_id in env.agents:
            action_str = "COOPERATE" if actions[agent_id] == 0 else "DEFECT"
            actions_log[agent_id].append(action_str)

        # Take a step in the environment using the chosen actions
        new_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Store rewards for visualization, ensuring indices are valid
        if len(env.agents) >= 2:
            reward_1 = rewards[env.agents[0]]
            reward_2 = rewards[env.agents[1]]
            rewards_player_1.append(reward_1)
            rewards_player_2.append(reward_2)
            combined_rewards.append(reward_1 + reward_2)

        # Print step info
        print(f"Step {step}:")
        for agent_id in env.agents:
            print(f"  Agent {agent_id} - Action: {actions[agent_id]}, Reward: {rewards[agent_id]}, Observation: {new_observations[agent_id]}")
            print(f"  Epsilon (exploration rate) for Agent {agent_id}: {agents[agent_id].epsilon}")
        print("")

        # Update observations for the next step
        observations = new_observations
        step += 1

env.close()

# Create a directory to save the figures if it doesn't exist
output_dir = 'figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot the rewards
plt.figure(figsize=(12, 6))
plt.scatter(range(len(rewards_player_1)), rewards_player_1, label='Player 1', s=10)
plt.scatter(range(len(rewards_player_2)), rewards_player_2, label='Player 2', s=10)
plt.scatter(range(len(combined_rewards)), combined_rewards, label='Combined Rewards', s=10)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Rewards for Both Players Over Time')
plt.legend()
plt.savefig(os.path.join(output_dir, "rewards.png"))

# Save actions to a file
with open(os.path.join(output_dir, 'actions_log.json'), 'w') as f:
    json.dump(actions_log, f)

print("Actions log saved to actions_log.json")

# Plot actions
fig, axs = plt.subplots(2, figsize=(12, 6), sharex=True)
fig.suptitle('Actions of Both Players Over Time')

# Convert actions to numerical values for plotting
actions_num_log = {agent_id: [0 if action == "COOPERATE" else 1 for action in actions_log[agent_id]] for agent_id in env.possible_agents}

# Plot actions for player 1
axs[0].scatter(range(len(actions_num_log[env.possible_agents[0]])), actions_num_log[env.possible_agents[0]], label='Player 1', s=10)
axs[0].set_ylabel('Action')
axs[0].set_yticks([0, 1])
axs[0].set_yticklabels(['COOPERATE', 'DEFECT'])
axs[0].legend()

# Plot actions for player 2
axs[1].scatter(range(len(actions_num_log[env.possible_agents[1]])), actions_num_log[env.possible_agents[1]], label='Player 2', s=10)
axs[1].set_xlabel('Step')
axs[1].set_ylabel('Action')
axs[1].set_yticks([0, 1])
axs[1].set_yticklabels(['COOPERATE', 'DEFECT'])
axs[1].legend()

fig.savefig(os.path.join(output_dir, "actions.png"))