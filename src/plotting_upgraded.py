import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    plt.savefig(os.path.join(folder_name, "average_rewards.png"))
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
        marker='o'
    )
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Average Reward")
    plt.title("Aggregated Average Reward Across All Runs")
    plt.legend(title="Player")
    plt.savefig(os.path.join(output_dir, "aggregated_rewards_individual.png"))
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
        marker='o'
    )
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Average Reward")
    plt.title("Aggregated Sum of Rewards Across All Runs")
    plt.savefig(os.path.join(output_dir, "aggregated_rewards_sum.png"))
    plt.close()


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

        for segment, (p1, p2) in enumerate(zip(p1_segment_rewards, p2_segment_rewards)):
            aggregated_data.append({"Segment": segment, "Reward": p1, "Player": "Player 1", "Run": i})
            aggregated_data.append({"Segment": segment, "Reward": p2, "Player": "Player 2", "Run": i})

        plot_individual_runs(folder_name, input_file)

    if aggregated_data:
        aggregated_df = pd.DataFrame(aggregated_data)
        plot_aggregated_data(aggregated_df, base_dir)


if __name__ == "__main__":
    BASE_DIR = "experiment_cnot_run02"
    NUM_FOLDERS = 15
    process_all_runs(BASE_DIR, NUM_FOLDERS)