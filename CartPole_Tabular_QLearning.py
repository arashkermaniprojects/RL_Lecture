"""
Introduction to Q-Learning with CartPole

This code demonstrates how to implement the Q-Learning algorithm on the CartPole-v1 environment from the Gymnasium library. 
Q-Learning is a reinforcement learning (RL) algorithm used to train an agent to maximize rewards by learning the best actions to take 
in different states.

The CartPole environment consists of a cart that can move left or right to balance a pole upright. The goal is to maximize the time 
the pole remains balanced.

This code includes:
1. State discretization: Converting continuous state values into discrete bins.
2. Q-Learning update formula for learning.
3. Epsilon-greedy policy for balancing exploration and exploitation.
4. Model saving/loading for reusability.
5. Separate training (without visualization) and testing (with visualization) phases.

Parameters, formulas, and every step are explained to help you understand the implementation.
"""
import gymnasium as gym
import numpy as np
import os
import pickle

# --- PARAMETERS ---
"""
n_bins: Number of bins for discretizing each state variable. 
        This determines how finely the state space is divided for Q-Learning.
    - Cart position (x): Horizontal position of the cart on the track.
    - Cart velocity (x_dot): Velocity of the cart.
    - Pole angle (theta): Angle of the pole with the vertical axis.
    - Pole angular velocity (theta_dot): Angular velocity of the pole.

alpha (Learning rate): Determines how much new information overrides old knowledge. 
                       Higher values make the model adapt faster but less stable.

gamma (Discount factor): Determines the importance of future rewards.
                         A value close to 1 means future rewards are highly valued.

epsilon (Exploration rate): Probability of choosing a random action during training.
                            Helps the agent explore new actions.

epsilon_min: The minimum value epsilon can decay to, ensuring some exploration persists.

epsilon_decay: The rate at which epsilon decreases after each episode to transition from exploration to exploitation.

training_episodes: Number of episodes for training the agent.

testing_episodes: Number of episodes for testing the agent.

model_file: File where the Q-table (learned policy) is saved/loaded.

use_existing_model: If True, skips training and loads the Q-table from the model file (if it exists).
"""

#n_bins = [Cart position,Cart velocity,Pole angle, Pole angular velocity]
n_bins = [200, 20, 200, 20]
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.02
epsilon_decay = 0.9999
training_episodes = 100000
testing_episodes = 10
model_file = "cartpole_q_table_10000.pkl"
use_existing_model =True

# --- ENVIRONMENT AND DISCRETIZATION ---
"""
The CartPole environment has a continuous state space. To apply tabular Q-Learning, we discretize this space into bins.
State variables include:
    - Cart position
    - Cart velocity
    - Pole angle
    - Pole angular velocity
Bins divide each variable's range into discrete intervals.
"""

env = gym.make("CartPole-v1")
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-3.5, 3.5)  # Adjust velocity bounds; initially (-inf, inf)
state_bounds[3] = (-3.5, 3.5)  # Adjust angular velocity bounds; initially (-inf, inf)
bins = [np.linspace(low, high, num=num_bins) for (low, high), num_bins in zip(state_bounds, n_bins)]
#linspace:
#len(bins) = 4
#len(bins[0]) = 200
#len(bins[1]) = 20
#len(bins[2]) = 200
#len(bins[3]) = 20
# Initialize the Q-table
n_actions = env.action_space.n  # Number of actions (2: move left or right)
q_table = np.zeros(tuple(len(bin_edges) + 1 for bin_edges in bins) + (n_actions,))

# --- HELPER FUNCTIONS ---
def discretize_state(state):
    """
    Converts continuous state variables into discrete indices based on bins.
    This makes the state compatible with the Q-table.
    """
    indices = []
    for i, (value, bin_edges) in enumerate(zip(state, bins)):
        index = np.digitize(value, bin_edges) - 1
        index = min(max(index, 0), len(bin_edges) - 1)
        indices.append(index)
    return tuple(indices)

def choose_action(state, epsilon):
    """
    Chooses an action using the epsilon-greedy policy.
    With probability epsilon, a random action is chosen (exploration).
    Otherwise, the best-known action (exploitation) is chosen.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(n_actions)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# --- TRAINING PHASE ---
if use_existing_model and os.path.exists(model_file):
    # Load the saved Q-table
    with open(model_file, "rb") as f:
        q_table = pickle.load(f)
    print("Model loaded from", model_file)
else:
    # Train the agent
    for episode in range(training_episodes):
        state, _ = env.reset()
        state = discretize_state(state)
        done = False
        total_reward = 0

        while not done:
            #choose the action using the epsilon-greedy policy
            action = choose_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)
            done = terminated or truncated

            # Update Q-value using Q-Learning formula:
            # Q(s, a) = Q(s, a) + alpha * [r + gamma * max(Q(s', a')) - Q(s, a)]
            #greedily select the action with the highest Q-value
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

            state = next_state
            total_reward += reward

        # Decay epsilon (reduce exploration over time)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Training Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Save the trained Q-table
    with open(model_file, "wb") as f:
        pickle.dump(q_table, f)
    print("Model saved to", model_file)

env.close()

# --- TESTING PHASE ---
"""
In the testing phase, the agent uses the learned policy (Q-table) to choose actions.
The environment is visualized to observe the agent's behavior.
"""
env = gym.make("CartPole-v1", render_mode="human")
for episode in range(testing_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state])  # Always exploit during testing
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)
        state = next_state
        total_reward += reward
        done = terminated or truncated

    print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
