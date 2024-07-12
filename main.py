import ipd_env
from agent import Agent

env = ipd_env.parallel_env(render_mode="human")
observations, infos = env.reset()

# Define state and action spaces
n_states = 2  
n_actions = 2


agents = {agent_id: Agent(n_states,n_actions) for agent_id in env.possible_agents}


step = 0
while env.agents:
    actions = {}
    for agent_id in env.agents:
        actions[agent_id] = agents[agent_id].choose_action(observations[agent_id])

    # Take a step in the environment
    new_observations, rewards, terminations, truncations, infos = env.step(actions)

    # Update Q-learning model (alle 25 steps)
    #for agent_id in env.agents:
         

    # Print step info
    print(f"Step {step}:")
    for agent_id in env.agents:
        print(f"  Agent {agent_id} - Action: {actions[agent_id]}, Reward: {rewards[agent_id]}, Observation: {new_observations[agent_id]}")
    print("")

    # Update observations
    observations = new_observations
    step += 1

env.close()