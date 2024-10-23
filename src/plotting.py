import os
import json
import matplotlib.pyplot as plt

def calculate_average_across_runs(input_file):
    # Load the detailed rewards data from the JSON file
    with open(input_file, 'r') as f:
        all_runs_avg_rewards = json.load(f)

    # Number of runs
    num_runs = len(all_runs_avg_rewards)

    # Assume all runs have the same number of segments
    num_segments = len(next(iter(all_runs_avg_rewards.values()))["Player_1_Average_Rewards"])

    # Initialize sums for each segment
    sum_rewards_p1 = [0] * num_segments
    sum_rewards_p2 = [0] * num_segments

    # Accumulate the rewards across all runs
    for rewards in all_runs_avg_rewards.values():
        player_1_avg_rewards = rewards["Player_1_Average_Rewards"]
        player_2_avg_rewards = rewards["Player_2_Average_Rewards"]

        for i in range(num_segments):
            sum_rewards_p1[i] += player_1_avg_rewards[i]
            sum_rewards_p2[i] += player_2_avg_rewards[i]

    # Calculate the average for each segment
    avg_rewards_p1 = [s / num_runs for s in sum_rewards_p1]
    avg_rewards_p2 = [s / num_runs for s in sum_rewards_p2]

    # Calculate the sum of the average rewards for both players
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
    input_file = os.path.join("results", "average_rewards_per_25_game_segment.json")
    output_dir = "results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot average rewards across all runs
    plot_average_rewards_across_all_runs(input_file, output_dir)