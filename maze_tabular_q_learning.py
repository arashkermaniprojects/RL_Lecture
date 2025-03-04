import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import truncnorm


# Define the Maze environment
class SimpleMaze:
    def __init__(self):
        self.maze = np.array([
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0]
        ])
        self.goal_position = (4, 4)
        self.start_position = (0, 0)
        self.current_position = self.start_position

    def take_action(self, action):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        next_position = (
            self.current_position[0] + actions[action][0],
            self.current_position[1] + actions[action][1]
        )

        if (0 <= next_position[0] < self.maze.shape[0] and
                0 <= next_position[1] < self.maze.shape[1] and
                self.maze[next_position] == 0):
            self.current_position = next_position
            if self.current_position == self.goal_position:
                return next_position, 10  # High reward for reaching the goal
            return next_position, -0.1  # Small negative reward for moving
        return self.current_position, -1  # Penalty for hitting a wall

    def reset(self):
        self.current_position = self.start_position
        return self.start_position
    

# Define the Q-Learning agent
class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9):
        self.q_table = np.zeros((maze.maze.shape[0], maze.maze.shape[1], 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def get_action(self, state, epsilon=0.1):

        randomvalue = np.random.uniform(0, 1)
        
        if randomvalue < epsilon:
            return np.random.randint(4)  # Explore
            
        return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action]
        )


# Visualization Function with `show_numbers` Flag
def render(maze, q_table, ax, agent_position, show_numbers=False):
    ax.clear()
    maze_copy = np.copy(maze.maze)
    maze_copy[maze.goal_position] = 3  # Mark the goal position
    cmap = plt.cm.plasma  # Use a perceptually uniform colormap
    norm = plt.Normalize(-1, 10)  # Normalize Q-values for the colormap

    # Unicode arrows for directions
    arrows = ['↑', '↓', '←', '→']

    # Draw the maze
    for i in range(maze.maze.shape[0]):
        for j in range(maze.maze.shape[1]):
            if maze.maze[i, j] == 1:  # Wall
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
            elif (i, j) == maze.goal_position:  # Goal
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))
            else:
                # Visualize arrows and optionally Q-values
                for action, value in enumerate(q_table[i, j]):
                    color = cmap(norm(value))  # Map Q-value to color for both numbers and arrows
                    if action == 0:  # Up
                        arrow_text = f"{arrows[action]}"  # Just the arrow
                        value_text = f"{value:.2f}"  # The number
                        ax.text(j + 0.7, i + 0.2, arrow_text, ha='center', fontsize=8, color=color)
                        if show_numbers:
                            ax.text(j + 0.7, i + 0.35, value_text, ha='center', fontsize=8, color=color)
                    elif action == 1:  # Down
                        arrow_text = f"{arrows[action]}"
                        value_text = f"{value:.2f}"
                        ax.text(j + 0.7, i + 0.8, arrow_text, ha='center', fontsize=8, color=color)
                        if show_numbers:
                            ax.text(j + 0.7, i + 0.65, value_text, ha='center', fontsize=8, color=color)
                    elif action == 2:  # Left
                        text = f"{arrows[action]} {value:.2f}" if show_numbers else arrows[action]
                        ax.text(j + 0.2, i + 0.5, text, va='center', fontsize=8, color=color)
                    elif action == 3:  # Right
                        text = f"{value:.2f} {arrows[action]}" if show_numbers else arrows[action]
                        ax.text(j + 0.8, i + 0.5, text, va='center', fontsize=8, color=color)

    # Add the agent (ball) at its current position
    ax.add_patch(Circle((agent_position[1] + 0.5, agent_position[0] + 0.5), 0.3, color='blue'))

    # Configure the grid
    ax.set_xlim(0, maze.maze.shape[1])
    ax.set_ylim(maze.maze.shape[0], 0)
    ax.set_aspect('equal')
    ax.axis('off')

# Training the agent with graphical visualization
maze = SimpleMaze()
agent = QLearningAgent(maze)
num_episodes = 10

fig, ax = plt.subplots(figsize=(6, 6))
plt.ion()  # Interactive mode for live updates

for episode in range(num_episodes):
    state = maze.reset()
    print(f"Episode {episode + 1}")
    render(maze, agent.q_table, ax, maze.current_position, show_numbers=True)  # Default: Show numbers
    plt.pause(1)  # Pause for visualization

    while state != maze.goal_position:
        action = agent.get_action(state)
        next_state, reward = maze.take_action(action)
        agent.update_q_table(state, action, next_state, reward)
        maze.current_position = next_state  # Update the agent's position
        state = next_state

        # Update the visualization
        render(maze, agent.q_table, ax, maze.current_position, show_numbers=True)
        plt.pause(0.5)  # Shorter pause for each step

    print(f"Episode {episode + 1} complete.\n")
    plt.pause(1)

plt.ioff()
plt.show()
