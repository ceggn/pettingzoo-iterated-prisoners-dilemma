import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pennylane as qml
from vqc_combined import VQC_Combined

sns.set_palette("colorblind")

NUM_GAMES_PER_SEGMENT = 25

def calculate_segment_rewards(rewards, num_segments):
    segment_rewards = []
    for segment in range(num_segments):
        start = segment * NUM_GAMES_PER_SEGMENT
        end = start + NUM_GAMES_PER_SEGMENT
        segment_rewards.append(np.mean(rewards[start:end]))
    return segment_rewards

def load_rewards(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    player_1_rewards = data["Player_1_Rewards"]
    player_2_rewards = data["Player_2_Rewards"]
    return player_1_rewards, player_2_rewards

def plot_individual_runs(folder_name, input_file):
    player_1_rewards, player_2_rewards = load_rewards(input_file)
    num_segments = len(player_1_rewards) // NUM_GAMES_PER_SEGMENT

    p1_segment_rewards = calculate_segment_rewards(player_1_rewards, num_segments)
    p2_segment_rewards = calculate_segment_rewards(player_2_rewards, num_segments)
    sum_segment_rewards = [p1 + p2 for p1, p2 in zip(p1_segment_rewards, p2_segment_rewards)]

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(num_segments), y=p1_segment_rewards, label="Player 1", marker='o')
    sns.lineplot(x=range(num_segments), y=p2_segment_rewards, label="Player 2", marker='o')
    sns.lineplot(x=range(num_segments), y=sum_segment_rewards, label="Sum of Rewards", marker='o', linestyle='--')
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward per Segment ({folder_name})")
    plt.legend()
    plt.savefig(os.path.join(folder_name, "average_rewards.pdf"))
    plt.close()

def plot_aggregated_data(dataframe, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot individual players
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=dataframe,
        x="Segment",
        y="Reward",
        hue="Player",
        errorbar="sd",
    )
    plt.axhline(y=1, color="red", linestyle="-", label="DEFECT")
    plt.axhline(y=3, color="green", linestyle="--", label="COOPERATE")
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Average Reward")
    plt.title("")
    plt.legend(title="Player")
    plt.savefig(os.path.join(output_dir, "aggregated_rewards_individual.pdf"))
    plt.close()

    # Compute sum of Player 1 and Player 2 rewards for each segment
    sum_rewards = dataframe.groupby(["Segment", "Run"]).sum().reset_index()
    sum_rewards["Player"] = "Sum of Players"
    sum_rewards = sum_rewards.rename(columns={"Reward": "Reward_Sum"})

    # Plot aggregated sum of rewards
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=sum_rewards,
        x="Segment",
        y="Reward_Sum",
        errorbar="sd",
    )
    plt.axhline(y=2, color="red", linestyle="-", label="DEFECT")
    plt.axhline(y=6, color="green", linestyle="--", label="COOPERATE")
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Average Reward")
    plt.title("")
    plt.legend(title="")
    plt.savefig(os.path.join(output_dir, "aggregated_rewards_sum.pdf"))
    plt.close()

def calculate_cooperation_rate(rewards, num_segments):
    """Calculate cooperation rate for each segment."""
    cooperation_rates = []
    for segment in range(num_segments):
        start = segment * NUM_GAMES_PER_SEGMENT
        end = start + NUM_GAMES_PER_SEGMENT
        segment_rewards = rewards[start:end]
        # Count cooperative actions (rewards associated with cooperation)
        cooperative_actions = sum(1 for reward in segment_rewards if reward in [3, 0])
        total_actions = len(segment_rewards)
        cooperation_rates.append(cooperative_actions / total_actions)
    return cooperation_rates

def plot_cooperation_rate(dataframe, output_dir):
    """Plot cooperation rate for players and aggregated data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=dataframe,
        x="Segment",
        y="Cooperation Rate",
        hue="Player",
        errorbar="sd",
        # marker='o'
    )
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Cooperation Rate")
    plt.title("Cooperation Rate Over Time")
    plt.legend(title="Player")
    plt.savefig(os.path.join(output_dir, "cooperation_rate_individual.pdf"))
    plt.close()

def calculate_defection_rate(rewards, num_segments):
    """Calculate defection rate for each segment."""
    defection_rates = []
    for segment in range(num_segments):
        start = segment * NUM_GAMES_PER_SEGMENT
        end = start + NUM_GAMES_PER_SEGMENT
        segment_rewards = rewards[start:end]
        # Count defecting actions (rewards associated with defection)
        defecting_actions = sum(1 for reward in segment_rewards if reward in [4, 1])
        total_actions = len(segment_rewards)
        defection_rates.append(defecting_actions / total_actions)
    return defection_rates

def plot_defection_rate(dataframe, output_dir):
    """Plot defection rate for players and aggregated data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=dataframe,
        x="Segment",
        y="Defection Rate",
        hue="Player",
        errorbar="sd",
        # marker='o'
    )
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Defection Rate")
    plt.title("Defection Rate Over Time")
    plt.legend(title="Player")
    plt.savefig(os.path.join(output_dir, "defection_rate_individual.pdf"))
    plt.close()

def visualize_vqc(output_filename):
    agents = ["agent1", "agent2"]
    agent_order = {"agent1": 0, "agent2": 1}
    observation_length = 2
    num_layers = 4
    action_space = 2

    vqc = VQC_Combined(agents, agent_order, observation_length, num_layers, action_space)

    x1 = [0, 1]
    x2 = [1, 0]
    weights = {agent: torch.randn(num_layers, observation_length, 3) for agent in agents}

    print("Generating quantum circuit visualization...")
    drawer_mpl = qml.draw_mpl(vqc.qnode)
    fig, ax = drawer_mpl(weights, x1, x2)
    plt.title("Quantum Circuit Visualization")
    fig.savefig(output_filename)
    print(f"Quantum circuit saved as {output_filename}")

def process_all_runs(base_dir, num_folders):
    aggregated_data = []

    for i in range(num_folders):
        folder_name = f"{base_dir}_{i}"
        input_file = os.path.join(folder_name, "complete_rewards.json")

        if not os.path.exists(input_file):
            print(f"Skipping {input_file}, file not found.")
            continue

        player_1_rewards, player_2_rewards = load_rewards(input_file)
        num_segments = len(player_1_rewards) // NUM_GAMES_PER_SEGMENT

        p1_segment_rewards = calculate_segment_rewards(player_1_rewards, num_segments)
        p2_segment_rewards = calculate_segment_rewards(player_2_rewards, num_segments)

        p1_cooperation_rate = calculate_cooperation_rate(player_1_rewards, num_segments)
        p2_cooperation_rate = calculate_cooperation_rate(player_2_rewards, num_segments)

        p1_defection_rate = calculate_defection_rate(player_1_rewards, num_segments)
        p2_defection_rate = calculate_defection_rate(player_2_rewards, num_segments)

        for segment, (p1, p2, p1_coop, p2_coop, p1_def, p2_def) in enumerate(zip(p1_segment_rewards, p2_segment_rewards, p1_cooperation_rate, p2_cooperation_rate, p1_defection_rate, p2_defection_rate)):
            aggregated_data.append({
                "Segment": segment,
                "Reward": p1,
                "Player": "Player 1",
                "Run": i,
                "Cooperation Rate": p1_coop,
                "Defection Rate": p1_def
            })
            aggregated_data.append({
                "Segment": segment,
                "Reward": p2,
                "Player": "Player 2",
                "Run": i,
                "Cooperation Rate": p2_coop,
                "Defection Rate": p2_def
            })

        plot_individual_runs(folder_name, input_file)

    if aggregated_data:
        aggregated_df = pd.DataFrame(aggregated_data)
        plot_aggregated_data(aggregated_df, base_dir)
        plot_cooperation_rate(aggregated_df, base_dir)
        plot_defection_rate(aggregated_df, base_dir)

    output_filename = os.path.join(base_dir, "vqc_circuit.png")
    visualize_vqc(output_filename)

if __name__ == "__main__":
    BASE_DIR = "results_5_new/experiment_cnot_run30"
    NUM_FOLDERS = 5
    process_all_runs(BASE_DIR, NUM_FOLDERS)