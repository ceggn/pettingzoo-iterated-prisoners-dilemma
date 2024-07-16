import ipd_env
from agent import Agent
import matplotlib.pyplot as plt
import json
import torch

# Initialize the environment
env = ipd_env.parallel_env(render_mode="human")
observations, infos = env.reset()

# Define state and action spaces
n_states = 2  # Number of possible states (example: 2 states)
n_actions = 2  # Number of possible actions (example: 2 actions)

# Initialize agents for each possible agent in the environment
agents = {agent_id: Agent(n_states, n_actions) for agent_id in env.possible_agents}

# Lists to store rewards
rewards_player_1 = []
rewards_player_2 = []

# Dictionaries to store actions at each step
actions_log = {agent_id: [] for agent_id in env.possible_agents}

step = 0
while env.agents:
    actions = {}
    
    # Each agent chooses an action based on their current state
    for agent_id in env.agents:
        actions[agent_id] = agents[agent_id].choose_action(observations[agent_id])

    # Log actions
    for agent_id in env.agents:
        actions_log[agent_id].append(actions[agent_id].item() if isinstance(actions[agent_id], torch.Tensor) else actions[agent_id])

    # Take a step in the environment using the chosen actions
    new_observations, rewards, terminations, truncations, infos = env.step(actions)

    # Store rewards for visualization, ensuring indices are valid
    if len(env.agents) >= 2:
        rewards_player_1.append(rewards[env.agents[0]])
        rewards_player_2.append(rewards[env.agents[1]])

    # Store the transitions in memory for each agent
    for agent_id in env.agents:
        agents[agent_id].store_transition(
            observations[agent_id],  # Current state
            actions[agent_id],       # Action taken
            rewards[agent_id],       # Reward received
            new_observations[agent_id],  # New state after action
            terminations[agent_id]   # Whether the episode has ended
        )

    # Update Q-learning model every 25 steps
    if step % 25 == 0:
        for agent_id in env.agents:
            agents[agent_id].train()

    # Print step info
    print(f"Step {step}:")
    for agent_id in env.agents:
        print(f"  Agent {agent_id} - Action: {actions[agent_id]}, Reward: {rewards[agent_id]}, Observation: {new_observations[agent_id]}")
        print(f"  Epsilon (exploration rate) for Agent {agent_id}: {agents[agent_id].epsilon}")
    print("")

    # Update observations for the next step
    observations = new_observations
    step += 1

    # Check if we should terminate the loop
    if all(terminations.values()) or all(truncations.values()):
        break

env.close()

# Plot the rewards
plt.figure(figsize=(12, 6))
plt.plot(rewards_player_1, label='Player 1')
plt.plot(rewards_player_2, label='Player 2')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Rewards for Both Players Over Time')
plt.legend()
plt.show()

# Save actions to a file
with open('actions_log.json', 'w') as f:
    json.dump(actions_log, f)

print("Actions log saved to actions_log.json")