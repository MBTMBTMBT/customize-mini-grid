import matplotlib.pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.distributions import Categorical
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from custom_env import CustomEnvFromFile
import imageio
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


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
    def __init__(self, capacity, alpha=0.8):
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


class PPOPolicyNetwork(nn.Module):
    def __init__(self, observation_channels, action_space):
        super(PPOPolicyNetwork, self).__init__()
        self.observation_channels = observation_channels
        self.action_space = action_space

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(observation_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
            nn.Flatten(),
        )

        # Fully connected layer for action logits
        self.fc_action = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

        # Fully connected layer for state value estimate
        self.fc_value = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        action_logits = self.fc_action(x)
        state_value = self.fc_value(x).squeeze(-1)  # Remove extra dimension for single value output
        return F.softmax(action_logits, dim=-1), state_value


# Define the DQN Agent
# PPO Agent Implementation
class PPOAgent:
    def __init__(self, observation_channels, action_space, lr=1e-3, gamma=0.99, clip_param=0.2, update_interval=4000,
                 epochs=10, device='cpu'):
        self.observation_channels = observation_channels
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.update_interval = update_interval
        self.epochs = epochs
        self.device = torch.device(device)

        self.policy = PPOPolicyNetwork(observation_channels, action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # These buffers store trajectories
        self.states = []
        self.actions = []
        self.state_values = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def act(self, state):
        state = state.float().to(self.device)
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        # self.states.append(state)
        # self.actions.append(action)
        self.log_probs.append(m.log_prob(action))
        self.state_values.append(state_value)
        return action.item()

    def calculate_returns(self, rewards, gamma, normalization=True):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if normalization:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        return returns

    def update(self):
        # Convert lists to tensors
        states = torch.stack(self.states).squeeze(1).to(self.device)
        actions = torch.tensor(self.actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)

        # Placeholder for rewards-to-go calculation; implement your method as needed
        rewards_to_go = self.calculate_returns(self.rewards, self.gamma).to(self.device)

        # Placeholder for advantage calculation; implement a more sophisticated method as needed
        advantages = rewards_to_go - torch.tensor(self.state_values).to(self.device).squeeze()

        # Calculate current log probs and state values for all stored states and actions
        probs, state_values = self.policy(states)
        dist = Categorical(probs)
        current_log_probs = dist.log_prob(actions)

        # Calculate the ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(current_log_probs - old_log_probs)

        # Calculate surrogate loss
        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()

        # Placeholder for value loss; consider using rewards_to_go for more accurate value updates
        value_loss = F.mse_loss(torch.squeeze(state_values), rewards_to_go.detach())

        # Take gradient step
        self.optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.state_values = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

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
        imageio.mimsave(filepath, images_with_actions, fps=3)


def preprocess_observation(obs: dict, rotate=False) -> torch.Tensor:
    """
    Preprocess the observation obtained from the environment to be suitable for the CNN.
    This function extracts, randomly rotates, and normalizes the 'image' part of the observation.

    :param obs: dict, The observation dictionary received from the environment.
                Expected to have a key 'image' containing the visual representation.
    :return: torch.Tensor, The normalized and randomly rotated image observation.
    """
    # Extract the 'image' array from the observation dictionary
    image_obs = obs['image']

    # Convert the numpy array to a PIL Image for rotation
    transform_to_pil = transforms.ToPILImage()
    pil_image = transform_to_pil(image_obs)

    # Convert the PIL Image back to a numpy array
    transform_to_tensor = transforms.ToTensor()

    # Randomly rotate the image
    # As the image is square, rotations of 0, 90, 180, 270 degrees will not require resizing
    if rotate:
        rotation_degrees = np.random.choice([0, 90, 180, 270])
        transform_rotate = transforms.RandomRotation([rotation_degrees, rotation_degrees])
        rotated_image = transform_rotate(pil_image)

        rotated_tensor = transform_to_tensor(rotated_image)
    else:
        rotated_tensor = transform_to_tensor(pil_image)

    # Normalize the tensor to [0, 1] (if not already normalized)
    rotated_tensor /= 255.0 if rotated_tensor.max() > 1.0 else 1.0

    # Change the order from (C, H, W) to (H, W, C)
    # rotated_tensor = rotated_tensor.permute(1, 2, 0)

    # Add a batch dimension
    rotated_tensor = rotated_tensor.unsqueeze(0)

    return rotated_tensor


def run_training(
    env: CustomEnvFromFile,
    agent: PPOAgent,
    episodes: int = 100,
    env_name: str = ""
) -> None:
    """
    Runs the training loop for a specified number of episodes using PPO.

    Args:
        env (CustomEnvFromFile): The environment instance where the agent will be trained.
        agent (PPOAgent): The agent to be trained with PPO.
        episodes (int): The total number of episodes to run for training.
        env_name (str): A name for the environment, used for saving outputs.

    Returns:
        None
    """
    for e in range(episodes):
        trajectory = []  # List to record each step for the GIF.
        obs, _ = env.reset()  # Reset the environment at the start of each episode.
        state = preprocess_observation(obs, rotate=True)  # Preprocess the observation for the agent.
        # state_img = obs['image']  # Store the original 'image' observation for visualization.
        # episode_states, episode_actions, episode_rewards, episode_log_probs = [], [], [], []

        for time in range(env.max_steps):
            action = agent.act(state)  # Agent selects an action based on the current state.
            next_obs, reward, terminated, truncated, info = env.step(action)  # Execute the action.
            agent.states.append(state)
            agent.actions.append(action)
            agent.rewards.append(float(reward))
            next_state = preprocess_observation(next_obs, rotate=True)  # Preprocess the new observation.
            trajectory.append((env.render(), action))  # Append the step for the GIF.

            done = terminated or truncated  # Check if the episode has ended.

            state = next_state  # Update the current state for the next iteration.

            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}")
                # Save the recorded trajectory as a GIF after each episode.
                agent.save_trajectory_as_gif(trajectory, agent.rewards, filename=env_name + f"_trajectory_{e}.gif")
                agent.update()
                break


if __name__ == "__main__":
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # List of environments to train on
    environment_files = [
        'short_corridor.txt',
        'long_corridor.txt',
        'small_maze.txt',
        'door_key.txt',
        # Add more file paths as needed
    ]

    # Training settings
    episodes_per_env = {
        'short_corridor.txt': 120,
        'long_corridor.txt': 120,
        'small_maze.txt': 120,
        'door_key.txt': 120,
        # Define episodes for more environments as needed
    }
    batch_size = 32

    for env_file in environment_files:
        # Initialize environment
        env = RGBImgObsWrapper(FullyObsWrapper(CustomEnvFromFile(txt_file_path=env_file, render_mode='rgb_array', size=None, max_steps=512)))
        image_shape = env.observation_space.spaces['image'].shape
        action_space = env.action_space.n

        # Initialize DQN agent for the current environment
        agent = PPOAgent(observation_channels=image_shape[-1], action_space=action_space, lr=1e-3, gamma=0.99, device=device)
        # Fetch the number of episodes for the current environment
        episodes = episodes_per_env.get(env_file, 100)  # Default to 100 episodes if not specified

        # Run training for the current environment
        print(f"Training on {env_file}")
        run_training(env, agent, episodes=episodes, env_name=env_file)
        print(f"Completed training on {env_file}")
