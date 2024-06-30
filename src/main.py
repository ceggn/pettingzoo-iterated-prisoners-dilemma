import os

import numpy as np

from agent import Agent
from ipd_env import env


def main():
    # Parameters
    buffer_len = 1000
    obs_window_length = 5
    n_actions = 2
    learning_rate = 0.003
    epsilon = 0.1
    n_batches = 5
    n_episodes = 1000
    checkpoint_dir = "./checkpoints"
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment
    environment = env(render_mode="human")
    
    # Initialize agents
    agent_1 = Agent(buffer_len, model="classical", obs_window_length=obs_window_length, n_actions=n_actions, learning_rate=learning_rate, epsilon=epsilon, n_batches=n_batches)
    agent_2 = Agent(buffer_len, model="classical", obs_window_length=obs_window_length, n_actions=n_actions, learning_rate=learning_rate, epsilon=epsilon, n_batches=n_batches)
    
    for episode in range(n_episodes):
        # Reset environment and get initial observations

        if environment.reset() is not None:
            observations, infos = environment.reset()
        
        done = False
        while not done:
            actions = {}
            # Agent 1 chooses an action based on its observation
            actions["player_0"] = agent_1.choose_action(observations["player_0"])
            # Agent 2 chooses an action based on its observation
            actions["player_1"] = agent_2.choose_action(observations["player_1"])
            
            # Environment steps
            next_observations, rewards, terminations, truncations, infos = environment.step(actions)
            
            # Store transitions in agents' memories
            agent_1.store_transition(observations["player_0"], actions["player_0"], rewards["player_0"], next_observations["player_0"], truncations["player_0"])
            agent_2.store_transition(observations["player_1"], actions["player_1"], rewards["player_1"], next_observations["player_1"], truncations["player_1"])
            
            # Train the agents
            agent_1.train()
            agent_2.train()
            
            # Update observations
            observations = next_observations
            
            # Check if the episode is done
            done = truncations["player_0"] or truncations["player_1"]
        
        # Print episode summary
        print(f"Episode {episode + 1}/{n_episodes} completed.")
    
    # Save models
    agent_1.save_model(checkpoint_dir, "agent_1_model.pth")
    agent_2.save_model(checkpoint_dir, "agent_2_model.pth")

if __name__ == "__main__":
    main()