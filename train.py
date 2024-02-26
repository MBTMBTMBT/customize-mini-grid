import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from minigrid.wrappers import FullyObsWrapper
from custom_env import CustomEnvFromFile
import imageio
import matplotlib.pyplot as plt
import os


ACTION_NAMES = {
    0: 'Turn Left',
    1: 'Turn Right',
    2: 'Move Forward',
    3: 'Pick Up',
    4: 'Drop',
    5: 'Toggle',
    6: 'Done'
}


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probabilities = prios ** self.alpha / np.sum(prios ** self.alpha)

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Avoid zero priority


# Define the DQN Agent
class DQNAgent:
    def __init__(self, action_space: int, lr: float = 1e-2, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1, epsilon_decay: float = 0.9995):
        """
        Initialize the DQN agent.

        :param action_space: int, Number of actions.
        :param lr: float, Learning rate for the optimizer.
        :param gamma: float, Discount factor for future rewards.
        :param epsilon_start: float, Starting value of epsilon for the epsilon-greedy policy.
        :param epsilon_end: float, Minimum value of epsilon.
        :param epsilon_decay: float, Decay rate of epsilon per episode.
        """
        self.action_space = action_space
        # self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = PrioritizedReplayBuffer(100000)

    def build_model(self) -> nn.Module:
        """
        Build the CNN model with adaptive pooling for processing variable size image inputs.
        """
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),  # Output size: 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space)
        )
        return model

    def act(self, state: np.ndarray) -> int:
        """
        Determine the action to take based on the current state.

        :param state: np.ndarray, The current state.
        :return: int, The action to take.
        """
        # Convert state to tensor and add batch dimension if it's not present
        state = torch.FloatTensor(state).unsqueeze(0) if state.ndim == 1 else torch.FloatTensor(state)
        if np.random.rand() <= self.epsilon:  # Explore
            return random.randrange(self.action_space)
        else:  # Exploit
            with torch.no_grad():
                action_values = self.model(state)
            return np.argmax(action_values.cpu().numpy())  # Choose action with highest Q-value

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.

        :param state: np.ndarray, The current state.
        :param action: int, The action taken.
        :param reward: float, The reward received.
        :param next_state: np.ndarray, The next state.
        :param done: bool, Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int):
        if len(self.memory.buffer) < batch_size:  # Ensure there are enough samples in the memory
            return

        minibatch, indices, weights = self.memory.sample(batch_size, beta=0.4)  # Use PER to sample
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors
        states = torch.stack(states).squeeze(1)
        next_states = torch.stack(next_states).squeeze(1)
        actions = torch.tensor(actions).view(-1, 1)
        rewards = torch.tensor(rewards).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.bool).view(-1, 1)
        weights = torch.tensor(weights, dtype=torch.float32)

        # Q-values for the current states
        Q_values = self.model(states).gather(1, actions)

        # Compute the expected Q-values
        next_Q_values = self.model(next_states).detach()
        max_next_Q_values = next_Q_values.max(1)[0].unsqueeze(1)
        expected_Q_values = rewards + (self.gamma * max_next_Q_values * (~dones))

        # Compute weighted loss
        loss = (weights * F.mse_loss(Q_values, expected_Q_values, reduction='none')).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the replay buffer
        new_priorities = abs(
            Q_values - expected_Q_values).detach().numpy() + 1e-5  # Adding a small constant to ensure no zero priority
        self.memory.update_priorities(indices, new_priorities)

        # Epsilon decay
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def create_image_with_action(self, image, action, step_number, reward):
        """
        Creates an image with the action text and additional details overlay.

        Parameters:
        - image: The image array in the correct format for matplotlib.
        - action: The action taken in this step.
        - step_number: The current step number.
        - reward: The reward received after taking the action.
        """
        # Convert action number to descriptive name and prepare the text
        action_text = ACTION_NAMES.get(action, f"Action {action}")
        details_text = f"Step: {step_number}, Reward: {reward}"

        # Normalize or convert the image if necessary
        image = image.astype(np.uint8)  # Ensure image is in uint8 format for display

        fig, ax = plt.subplots()
        ax.imshow(image)
        # Position the action text
        ax.text(0.5, -0.1, action_text, color='white', transform=ax.transAxes,
                ha="center", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        # Position the details text (step number and reward)
        ax.text(0.5, -0.15, details_text, color='white', transform=ax.transAxes,
                ha="center", fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
        ax.axis('off')

        # Convert the Matplotlib figure to an image array and close the figure to free memory
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        return img_array

    def save_trajectory_as_gif(self, trajectory, rewards, folder="trajectories", filename="trajectory.gif"):
        """
        Saves the trajectory as a GIF in a specified folder, including step numbers and rewards.

        Parameters:
        - trajectory: List of tuples, each containing (image, action).
        - rewards: List of rewards for each step in the trajectory.
        """
        # Ensure the target folder exists
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        images_with_actions = [self.create_image_with_action(img, action, step_number, rewards[step_number])
                               for step_number, (img, action) in enumerate(trajectory)]
        imageio.mimsave(filepath, images_with_actions, fps=1.5)


def preprocess_observation(obs: dict) -> torch.Tensor:
    """
    Preprocess the observation obtained from the environment to be suitable for the CNN.

    This function extracts and normalizes the 'image' part of the observation.

    :param obs: dict, The observation dictionary received from the environment.
                Expected to have a key 'image' containing the visual representation.
    :return: torch.Tensor, The normalized image observation.
    """
    # Extract the 'image' array from the observation dictionary
    image_obs = obs['image']
    # Normalize the image to [0, 1]
    image_obs = image_obs / 255.0
    # Convert to PyTorch tensor and add a batch dimension
    return torch.FloatTensor(image_obs).permute(2, 0, 1).unsqueeze(0)  # Change to (C, H, W) and add batch dimension


if __name__ == "__main__":
    # Create an environment instance wrapped to provide fully observable states
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path='simple_test.txt', render_mode='rgb_array'))

    # Determine the shape of the 'image' observation space to calculate the state space
    # The 'image' space is expected to be a 3D array (e.g., width x height x RGB channels)
    image_shape = env.observation_space.spaces['image'].shape  # type: tuple

    # Retrieve the number of possible actions from the environment's action space
    action_space = env.action_space.n  # type: int

    # Initialize the DQN agent with the state space and action space dimensions
    agent = DQNAgent(action_space)

    # Set the number of episodes to run the training for
    episodes = 100  # type: int
    # Set the batch size for experience replay; using 1 for this example
    batch_size = 32  # type: int

    # Main training loop
    for e in range(episodes):
        trajectory = []  # Initialize the trajectory list to record each step for the GIF
        rewards = []  # Initialize the rewards list for the episode
        obs, _ = env.reset()  # Reset the environment at the start of each episode and preprocess the initial observation
        state = preprocess_observation(obs)
        # For GIF creation, keep the original 'image' observation for visualization
        state_img = obs['image']  # Assuming 'image' is the RGB image of the state

        # Iterate through steps within the episode
        for time in range(env.max_steps):
            action = agent.act(state)  # Agent selects an action based on the current state
            next_obs, reward, terminated, truncated, info = env.step(action)  # Environment executes the action and returns the next observation and other details
            next_state = preprocess_observation(next_obs)
            next_state_img = next_obs['image']  # Keep the raw image of the next state for GIF creation
            rewards.append(reward)  # Append the received reward to the rewards list

            # Append the current state's image and action to the trajectory
            trajectory.append((env.render(), action))

            # Determine if the episode has ended, either by termination or truncation
            done = terminated or truncated

            # Adjusted to use the new method of adding experiences with an error term
            agent.memory.add(state, action, reward, next_state, terminated or truncated)

            # Update the current state to be the next state
            state = next_state
            state_img = next_state_img  # Update state_img for the next iteration

            # If the episode is done, print the episode stats and break from the loop
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break

            # If there are enough transitions in memory, perform a replay step
            if len(agent.memory.buffer) > batch_size:
                agent.replay(batch_size)

        # After each episode, save the recorded trajectory as a GIF
        agent.save_trajectory_as_gif(trajectory, rewards, filename=f"trajectory_{e}.gif")
