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
    plt.xlabel("Episodes (25-Game Segment)", fontsize=18)
    plt.ylabel("Average Reward", fontsize=18)
    plt.title(f"Average Reward per Segment ({folder_name})", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(title="Player", fontsize=15, title_fontsize=18)
    plt.savefig(os.path.join(folder_name, "average_rewards.pdf"))
    plt.close()

def calculate_cooperation_rate(rewards, num_segments):
    cooperation_rates = []
    for segment in range(num_segments):
        start = segment * NUM_GAMES_PER_SEGMENT
        end = start + NUM_GAMES_PER_SEGMENT
        segment_rewards = rewards[start:end]
        cooperative_actions = sum(1 for reward in segment_rewards if reward in [3, 0])
        total_actions = len(segment_rewards)
        cooperation_rates.append(cooperative_actions / total_actions)
    return cooperation_rates

def calculate_defection_rate(rewards, num_segments):
    defection_rates = []
    for segment in range(num_segments):
        start = segment * NUM_GAMES_PER_SEGMENT
        end = start + NUM_GAMES_PER_SEGMENT
        segment_rewards = rewards[start:end]
        defecting_actions = sum(1 for reward in segment_rewards if reward in [4, 1])
        total_actions = len(segment_rewards)
        defection_rates.append(defecting_actions / total_actions)
    return defection_rates

def process_multiple_experiments(base_dirs, num_folders):
    aggregated_data = []

    for base_dir, experiment_name in base_dirs.items():  
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

            for segment, (p1, p2, p1_coop, p2_coop, p1_def, p2_def) in enumerate(zip(
                    p1_segment_rewards, p2_segment_rewards,
                    p1_cooperation_rate, p2_cooperation_rate,
                    p1_defection_rate, p2_defection_rate)):
                aggregated_data.append({
                    "Segment": segment,
                    "Reward": p1,
                    "Player": "Player 1",
                    "Run": i,
                    "Experiment": experiment_name, 
                    "Cooperation Rate": p1_coop,
                    "Defection Rate": p1_def
                })
                aggregated_data.append({
                    "Segment": segment,
                    "Reward": p2,
                    "Player": "Player 2",
                    "Run": i,
                    "Experiment": experiment_name,  
                    "Cooperation Rate": p2_coop,
                    "Defection Rate": p2_def
                })

            plot_individual_runs(folder_name, input_file)

    if aggregated_data:
        aggregated_df = pd.DataFrame(aggregated_data)
        plot_aggregated_experiments(aggregated_df, list(base_dirs.keys())[0])  

def plot_aggregated_experiments(dataframe, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=dataframe,
        x="Segment",
        y="Reward",
        hue="Experiment",
        style="Player",
        errorbar=None,
        # marker='o'
    )
    plt.xlabel("Episodes (25-Game Segment)", fontsize=18)
    plt.ylabel("Average Reward", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(title="Experiment & Player", fontsize=15, title_fontsize=18)
    plt.savefig(os.path.join(output_dir, "aggregated_rewards_experiments.pdf"))
    plt.close()

    sum_rewards = dataframe.groupby(["Segment", "Experiment", "Run"]).sum().reset_index()
    sum_rewards["Player"] = "Sum of Players"
    sum_rewards = sum_rewards.rename(columns={"Reward": "Reward_Sum"})

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=sum_rewards,
        x="Segment",
        y="Reward_Sum",
        hue="Experiment",
        errorbar=None,
        # marker='o'
    )
    plt.axhline(y=2, color="red", linestyle="-", label="DEFECT")  
    plt.axhline(y=6, color="green", linestyle="--", label="COOPERATE")  
    plt.xlabel("Episodes (25-Game Segment)", fontsize=18)
    plt.ylabel("Average Reward", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(title="Experiment", fontsize=15, title_fontsize=18)
    plt.savefig(os.path.join(output_dir, "aggregated_rewards_sum_experiments.pdf"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=dataframe,
        x="Segment",
        y="Cooperation Rate",
        hue="Experiment",
        style="Player",
        errorbar=None,
        # marker='o'
    )
    plt.xlabel("Episodes (25-Game Segment)", fontsize=18)
    plt.ylabel("Cooperation Rate", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(title="Experiment & Player", fontsize=15, title_fontsize=18)
    plt.savefig(os.path.join(output_dir, "cooperation_rate_experiments.pdf"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=dataframe,
        x="Segment",
        y="Defection Rate",
        hue="Experiment",
        style="Player",
        errorbar=None,
        # marker='o'
    )
    plt.xlabel("Episodes (25-Game Segment)", fontsize=18)
    plt.ylabel("Defection Rate", fontsize=18)
    plt.title("Defection Rate Comparison Across Experiments", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(title="Experiment & Player", fontsize=15, title_fontsize=18)
    plt.savefig(os.path.join(output_dir, "defection_rate_experiments.pdf"))
    plt.close()


if __name__ == "__main__":
    EXPERIMENTS = {
        "results_5_new/experiment_baseline": "Baseline",
        "results_5_new/experiment_cnot_run09": "Experiment 9, Bidirectional Entanglement",
        "results_5_new/experiment_cnot_run22": "Experiment 22, Symmetric Swap",
        "results_5_new/experiment_cnot_run31": "Experiment 31, Modular CZ Alignment",
        "results_5_new/experiment_cnot_run32": "Experiment 32, Controlled SWAP",
    }
    NUM_FOLDERS = 5

    process_multiple_experiments(EXPERIMENTS, NUM_FOLDERS)