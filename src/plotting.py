import os
import json
import numpy as np
import matplotlib.pyplot as plt

NUM_GAMES_PER_SEGMENT = 25

def calculate_average_across_runs(input_file):
    # Load the complete rewards data from the JSON file
    with open(input_file, 'r') as f:
        all_runs_rewards = json.load(f)

    num_runs = len(all_runs_rewards)

    # Number of segments based on the assumption that each run has a consistent length
    total_games = len(next(iter(all_runs_rewards.values()))["Player_1_Rewards"])
    num_segments = total_games // NUM_GAMES_PER_SEGMENT

    # Initialize lists to accumulate rewards across runs
    sum_rewards_p1 = [0] * num_segments
    sum_rewards_p2 = [0] * num_segments

    # Accumulate the rewards for each segment across all runs
    for rewards in all_runs_rewards.values():
        player_1_rewards = rewards["Player_1_Rewards"]
        player_2_rewards = rewards["Player_2_Rewards"]

        for segment in range(num_segments):
            start = segment * NUM_GAMES_PER_SEGMENT
            end = start + NUM_GAMES_PER_SEGMENT

            # Calculate the sum for each segment
            sum_rewards_p1[segment] += np.mean(player_1_rewards[start:end])
            sum_rewards_p2[segment] += np.mean(player_2_rewards[start:end])

    # Calculate the average for each segment
    avg_rewards_p1 = [s / num_runs for s in sum_rewards_p1]
    avg_rewards_p2 = [s / num_runs for s in sum_rewards_p2]
    sum_avg_rewards = [avg_rewards_p1[i] + avg_rewards_p2[i] for i in range(num_segments)]

    return avg_rewards_p1, avg_rewards_p2, sum_avg_rewards

def plot_average_rewards_across_all_runs(input_file, output_dir):
    # Calculate the average rewards across all runs
    avg_rewards_p1, avg_rewards_p2, sum_avg_rewards = calculate_average_across_runs(input_file)

    # Plot the average rewards per segment across all runs
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards_p1, label="Player 1 - Average Reward Across Runs", marker='o')
    plt.plot(avg_rewards_p2, label="Player 2 - Average Reward Across Runs", marker='o')
    plt.plot(sum_avg_rewards, label="Sum of Player 1 and Player 2 - Average Reward", marker='o', linestyle='--')
    plt.xlabel("Episodes (25-Game Segment)")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per 25-Game Segment (Across All Runs)")
    plt.legend()

    # Save the plot
    avg_plot_filename = "average_rewards_across_all_runs.png"
    plt.savefig(os.path.join(output_dir, avg_plot_filename))
    plt.show()
    plt.close()

if __name__ == "__main__":


    for i in range(15):  
        input_file = os.path.join(f"results_truncationfix_{i}", "complete_rewards.json")
        
        output_dir = f"results_truncationfix_{i}"
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Plot average rewards across all runs
        plot_average_rewards_across_all_runs(input_file, output_dir)