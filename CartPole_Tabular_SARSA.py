import gymnasium as gym
import numpy as np
import os
import pickle

# --- PARAMETERS ---
# n_bins: Number of bins for discretizing each state variable. This determines how finely the state space is divided for SARSA.
# - Cart position (x): Horizontal position of the cart on the track.
# - Cart velocity (x_dot): Velocity of the cart.
# - Pole angle (theta): Angle of the pole with the vertical axis.
# - Pole angular velocity (theta_dot): Angular velocity of the pole.
#
# alpha (Learning rate): Determines how much new information overrides old knowledge.
# gamma (Discount factor): Determines the importance of future rewards.
# epsilon (Exploration rate): Probability of choosing a random action during training.
# epsilon_min: The minimum value epsilon can decay to, ensuring some exploration persists.
# epsilon_decay: The rate at which epsilon decreases after each episode to transition from exploration to exploitation.
# training_episodes: Number of episodes for training the agent.
# testing_episodes: Number of episodes for testing the agent.
# model_file: File where the Q-table (learned policy) is saved/loaded.
# use_existing_model: If True, skips training and loads the Q-table from the model file (if it exists).
n_bins = [200, 20, 200, 20]
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.2
epsilon_decay = 0.9999
training_episodes = 100000
testing_episodes = 10
model_file = "cartpole_sarsa_table.pkl"
use_existing_model = False

# --- ENVIRONMENT AND DISCRETIZATION ---
# The CartPole environment has a continuous state space. To apply tabular SARSA, we discretize this space into bins.
# State variables include:
# - Cart position
# - Cart velocity
# - Pole angle
# - Pole angular velocity
# Bins divide each variable's range into discrete intervals.
env = gym.make("CartPole-v1")
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = (-3.5, 3.5)  # Adjust velocity bounds
state_bounds[3] = (-3.5, 3.5)  # Adjust angular velocity bounds
bins = [np.linspace(low, high, num=num_bins) for (low, high), num_bins in zip(state_bounds, n_bins)]

# Initialize the Q-table
# The Q-table stores the expected rewards for each state-action pair.
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
        index = min(max(index, 0), len(bin_edges) - 1)  # Ensure index is within valid range
        indices.append(index)
    return tuple(indices)

def choose_action(state, epsilon):
    """
    Chooses an action using the epsilon-greedy policy.
    With probability epsilon, a random action is chosen (exploration).
    Otherwise, the best-known action (exploitation) is chosen based on the Q-table.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(n_actions)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# --- TRAINING PHASE ---
if use_existing_model and os.path.exists(model_file):
    # Load the saved Q-table if it exists
    with open(model_file, "rb") as f:
        q_table = pickle.load(f)
    print("Model loaded from", model_file)
else:
    # Train the agent using SARSA
    for episode in range(training_episodes):
        state, _ = env.reset()  # Reset the environment to start a new episode
        state = discretize_state(state)  # Discretize the continuous state
        action = choose_action(state, epsilon)  # Choose an action using epsilon-greedy policy
        done = False
        total_reward = 0

        while not done:
            # Perform the selected action and observe the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state)  # Discretize the next state
            next_action = choose_action(next_state, epsilon)  # Choose the next action using epsilon-greedy policy
            done = terminated or truncated

            # Update Q-value using SARSA formula:
            # Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])

            # Transition to the next state and action
            state, action = next_state, next_action
            total_reward += reward

        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Training Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Save the trained Q-table for future use
    with open(model_file, "wb") as f:
        pickle.dump(q_table, f)
    print("Model saved to", model_file)

env.close()

# --- TESTING PHASE ---
# In the testing phase, the agent uses the learned policy (Q-table) to choose actions.
# The environment is visualized to observe the agent's behavior.
env = gym.make("CartPole-v1", render_mode="human")
for episode in range(testing_episodes):
    state, _ = env.reset()  # Reset the environment to start a new episode
    state = discretize_state(state)  # Discretize the continuous state
    done = False
    total_reward = 0

    while not done:
        # Always exploit the learned policy during testing
        action = np.argmax(q_table[state])  # Select the action with the highest Q-value
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)  # Discretize the next state
        state = next_state
        total_reward += reward
        done = terminated or truncated

    print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
