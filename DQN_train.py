import matplotlib.pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from custom_env import CustomEnvFromFile
import imageio
import matplotlib.pyplot as plt
import os
import copy


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


# Define the DQN Agent
class DQNAgent:
    def __init__(self, observation_channels: int, action_space: int, lr: float = 1e-2, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1, epsilon_decay: float = 0.95, device: str = 'cpu'):
        """
        Initialize the DQN agent.

        :param action_space: int, Number of actions.
        :param lr: float, Learning rate for the optimizer.
        :param gamma: float, Discount factor for future rewards.
        :param epsilon_start: float, Starting value of epsilon for the epsilon-greedy policy.
        :param epsilon_end: float, Minimum value of epsilon.
        :param epsilon_decay: float, Decay rate of epsilon per episode.
        """
        self.obs_channels = observation_channels
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.model = self.build_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = PrioritizedReplayBuffer(1000)
        self.target_model = copy.deepcopy(self.model)  # Assuming self.model is your online network
        self.target_model.to(self.device)

    def build_model(self) -> nn.Module:
        """
        Build the CNN model with adaptive pooling for processing variable size image inputs.
        """
        model = nn.Sequential(
            nn.Conv2d(self.obs_channels, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
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
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0) if state.ndim == 1 else torch.FloatTensor(state).to(self.device)
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
        # Function to pad images within each batch to the same size
        def pad_images(images):
            # Find the max width and height in the batch
            max_height = max(image.shape[1] for image in images)
            max_width = max(image.shape[2] for image in images)

            # Pad images to the max width and height
            padded_images = [F.pad(image, (0, max_width - image.shape[2], 0, max_height - image.shape[1])) for image in
                             images]
            return torch.stack(padded_images)

        if len(self.memory.buffer) < batch_size:  # Ensure there are enough samples in the memory
            return

        minibatch, indices, weights = self.memory.sample(batch_size, beta=0.4)  # Use PER to sample
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Preprocess and pad states and next_states
        states = pad_images([torch.FloatTensor(state) for state in states]).squeeze(1).to(self.device)
        next_states = pad_images([torch.FloatTensor(state) for state in next_states]).squeeze(1).to(
            self.device)

        actions = torch.tensor(actions, device=self.device).view(-1, 1)
        rewards = torch.tensor(rewards, device=self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device).view(-1, 1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Online network estimates which action is best in the next state
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)

        # Target network evaluates the Q-value of the action selected by the online network
        next_Q_values = self.target_model(next_states).gather(1, next_actions).detach()
        expected_Q_values = rewards + self.gamma * next_Q_values * (~dones)

        # Compute weighted loss
        Q_values = self.model(states).gather(1, actions)
        loss = (weights * F.mse_loss(Q_values, expected_Q_values, reduction='none')).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the replay buffer
        new_priorities = abs(
            Q_values - expected_Q_values).detach().cpu().numpy() + 1e-5  # Adding a small constant to ensure no zero priority
        self.memory.update_priorities(indices, new_priorities)

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
    agent: DQNAgent,
    episodes: int = 100,
    batch_size: int = 32,
    target_update_interval: int = 10,
    env_name: str = ""
) -> None:
    """
    Runs the training loop for a specified number of episodes.

    Args:
        env (CustomEnvFromFile): The environment instance where the agent will be trained.
        agent (DQNAgent): The agent to be trained.
        episodes (int): The total number of episodes to run for training.
        batch_size (int): The batch size for experience replay during training.

    Returns:
        None
    """
    for e in range(episodes):
        # if e % target_update_interval == 0:
        #     agent.target_model.load_state_dict(agent.model.state_dict())

        trajectory = []  # List to record each step for the GIF.
        rewards = []  # List to store rewards received each step of the episode.
        obs, _ = env.reset()  # Reset the environment at the start of each episode.
        state = preprocess_observation(obs)  # Preprocess the observation for the agent.
        state_img = obs['image']  # Store the original 'image' observation for visualization.

        for time in range(env.max_steps):
            action = agent.act(state)  # Agent selects an action based on the current state.
            next_obs, reward, terminated, truncated, info = env.step(action)  # Execute the action.
            next_state = preprocess_observation(next_obs)  # Preprocess the new observation.
            rewards.append(reward)  # Append the received reward to the rewards list.
            trajectory.append((env.render(), action))  # Append the step for the GIF.

            done = terminated or truncated  # Check if the episode has ended.
            rotated_next_state = preprocess_observation(next_obs, rotate=True)  # randomly rotate the image before adding into the buffer
            agent.memory.add(state, action, reward, rotated_next_state, done)  # Add experience to the buffer.

            state = next_state  # Update the current state for the next iteration.

            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break

            if len(agent.memory.buffer) > batch_size:
                agent.replay(batch_size)  # Perform a training step if enough experiences are in the buffer.

        # Epsilon decay after each episode.
        if agent.epsilon > agent.epsilon_end:
            agent.epsilon *= agent.epsilon_decay

        # Save the recorded trajectory as a GIF after each episode.
        agent.save_trajectory_as_gif(trajectory, rewards, filename=env_name+f"_trajectory_{e}.gif")


if __name__ == "__main__":
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # List of environments to train on
    environment_files = [
        'simple_test_corridor.txt',
        'simple_test_corridor_long.txt',
        'simple_test_maze_small.txt',
        'simple_test_door_key.txt',
        # Add more file paths as needed
    ]

    # Training settings
    episodes_per_env = {
        'simple_test_corridor.txt': 120,
        'simple_test_corridor_long.txt': 120,
        'simple_test_maze_small.txt': 120,
        'simple_test_door_key.txt': 120,
        # Define episodes for more environments as needed
    }
    batch_size = 32

    for env_file in environment_files:
        # Initialize environment
        env = RGBImgObsWrapper(FullyObsWrapper(CustomEnvFromFile(txt_file_path=env_file, render_mode='rgb_array', size=None, max_steps=512)))
        image_shape = env.observation_space.spaces['image'].shape
        action_space = env.action_space.n

        # Initialize DQN agent for the current environment
        agent = DQNAgent(observation_channels=image_shape[-1], action_space=action_space, lr=1e-4, gamma=0.99, device=device)
        agent.memory = PrioritizedReplayBuffer(2**16)  # Use a large buffer size

        # Reset agent's exploration rate for each new environment
        agent.epsilon = 0.75
        agent.epsilon_decay = 0.99

        # Fetch the number of episodes for the current environment
        episodes = episodes_per_env.get(env_file, 100)  # Default to 100 episodes if not specified

        # Run training for the current environment
        print(f"Training on {env_file}")
        run_training(env, agent, episodes=episodes, batch_size=batch_size, env_name=env_file)
        print(f"Completed training on {env_file}")
