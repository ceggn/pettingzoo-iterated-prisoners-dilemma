# Iterated Prisoner's Dilemma (IPD) with Q-learning Agents

This project simulates the Iterated Prisoner's Dilemma (IPD) game using Q-learning agents. The agents learn to cooperate or defect based on the rewards received over multiple episodes. The project visualizes the rewards and actions of the agents over time.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Simulation Details](#simulation-details)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ipd-qlearning.git
    cd ipd-qlearning
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the simulation:

1. Ensure the virtual environment is activated.
2. Run the main script:
    ```bash
    python main.py
    ```

This will run the simulation, train the agents, and generate plots of the rewards and actions over time.

## Project Structure

- `main.py`: The main script to run the simulation.
- `ipd_env.py`: Defines the IPD environment using the PettingZoo library.
- `agent.py`: Defines the Q-learning agent.
- `model.py`: Defines the Q-learning model used by the agents.
- `requirements.txt`: Lists the dependencies for the project.
- `figures/`: Directory where the plots are saved.
- `README.md`: Project documentation.

## Simulation Details

The simulation involves the following steps:

1. **Environment Initialization**: The IPD environment is created with two agents.
2. **Agent Initialization**: Each agent is initialized with Q-learning parameters.
3. **Simulation Loop**: The environment is reset and agents take actions based on their Q-learning policies. Rewards are recorded, and the Q-values are updated.
4. **Training**: The agents' Q-learning models are trained using experience replay.
5. **Visualization**: Rewards and actions are plotted and saved.

### Parameters

- `n_states`: Number of possible states.
- `n_actions`: Number of possible actions.
- `n_games`: Number of games/episodes.
- `alpha`: Learning rate.
- `epsilon`: Exploration rate for epsilon-greedy action selection.
- `gamma`: Discount factor for future rewards.
- `epsilon_decay`: Decay rate for `epsilon`.
- `epsilon_min`: Minimum value for `epsilon`.

## Results

The simulation generates the following results:

- **Rewards Plot**: Shows the rewards for both players and their combined rewards over time (`rewards.png`).
- **Actions Log**: A JSON file logging the actions of both players (`actions_log.json`).
- **Actions Plot**: Shows the actions (cooperate or defect) of both players over time (`actions.png`).

