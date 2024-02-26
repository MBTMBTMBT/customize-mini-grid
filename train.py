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


# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_space: int, action_space: int, lr: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.995):
        """
        Initialize the DQN agent.

        :param state_space: int, Dimensionality of the state space.
        :param action_space: int, Number of actions.
        :param lr: float, Learning rate for the optimizer.
        :param gamma: float, Discount factor for future rewards.
        :param epsilon_start: float, Starting value of epsilon for the epsilon-greedy policy.
        :param epsilon_end: float, Minimum value of epsilon.
        :param epsilon_decay: float, Decay rate of epsilon per episode.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def build_model(self) -> nn.Module:
        """
        Build the neural network model.

        :return: Sequential model of the neural network.
        """
        model = nn.Sequential(
            nn.Linear(self.state_space, 64),  # Input layer to hidden layer 1
            nn.ReLU(),  # Activation function for hidden layer 1
            nn.Linear(64, 64),  # Hidden layer 1 to hidden layer 2
            nn.ReLU(),  # Activation function for hidden layer 2
            nn.Linear(64, self.action_space)  # Hidden layer 2 to output layer
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
        """
        Train the model using a mini-batch of transitions from the replay buffer.

        :param batch_size: int, The size of the mini-batch to train on.
        """
        if len(self.memory) < batch_size:  # Ensure there are enough samples in the memory
            return

        minibatch = random.sample(self.memory, batch_size)  # Sample a mini-batch
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors
        states = torch.FloatTensor(states).squeeze(1)
        actions = torch.LongTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states).squeeze(1)
        dones = torch.BoolTensor(dones)

        # Q-values for the current states
        Q_values = self.model(states).gather(1, actions)

        # Compute the expected Q-values
        next_Q_values = self.model(next_states).detach()
        max_next_Q_values = next_Q_values.max(1)[0].unsqueeze(1)
        expected_Q_values = rewards + (self.gamma * max_next_Q_values * (~dones))

        # Compute loss
        loss = F.mse_loss(Q_values, expected_Q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def create_image_with_action(self, image, action):
        """
        Creates an image with the action text overlay.
        Assumes image is in the correct format for matplotlib (e.g., uint8, normalized if needed).
        """
        # Convert action number to descriptive name
        action_text = ACTION_NAMES.get(action, f"Action {action}")

        # Normalize or convert the image if necessary
        image = image.astype(np.uint8)  # Ensure image is in uint8 format for display

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.text(0.5, -0.1, action_text, color='white', transform=ax.transAxes,
                ha="center", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')

        # Convert the Matplotlib figure to an image array and close the figure to free memory
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        return img_array

    def save_trajectory_as_gif(self, trajectory, folder="trajectories", filename="trajectory.gif"):
        """
        Saves the trajectory as a GIF in a specified folder.
        """
        # Ensure the target folder exists
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)

        images_with_actions = [self.create_image_with_action(img, action) for img, action in trajectory]
        imageio.mimsave(filepath, images_with_actions, fps=1)


def preprocess_observation(obs: dict) -> np.ndarray:
    """
    Preprocess the observation obtained from the environment.

    This function extracts the 'image' part of the observation, which is assumed
    to be a grid or visual representation of the environment's current state, and
    flattens it into a 1D numpy array.

    :param obs: dict, The observation dictionary received from the environment.
                Expected to have a key 'image' containing the visual representation.
    :return: np.ndarray, The flattened image observation.
    """
    # Extract the 'image' array from the observation dictionary
    image_obs = obs['image']
    # Flatten the image array to create a 1D representation of the state
    return np.ravel(image_obs)


if __name__ == "__main__":
    # Create an environment instance wrapped to provide fully observable states
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path='simple_test.txt', render_mode='rgb_array'))

    # Determine the shape of the 'image' observation space to calculate the state space
    # The 'image' space is expected to be a 3D array (e.g., width x height x RGB channels)
    image_shape = env.observation_space.spaces['image'].shape  # type: tuple
    # Calculate the total number of elements in the flattened 'image' array
    state_space = np.prod(image_shape)  # type: int

    # Retrieve the number of possible actions from the environment's action space
    action_space = env.action_space.n  # type: int

    # Initialize the DQN agent with the state space and action space dimensions
    agent = DQNAgent(state_space, action_space)

    # Set the number of episodes to run the training for
    episodes = 100  # type: int
    # Set the batch size for experience replay; using 1 for this example
    batch_size = 1  # type: int

    # Main training loop
    for e in range(episodes):
        trajectory = []  # Initialize the trajectory list to record each step for the GIF
        obs, _ = env.reset()  # Reset the environment at the start of each episode and preprocess the initial observation
        state = preprocess_observation(obs)
        # For GIF creation, keep the original 'image' observation for visualization
        state_img = obs['image']  # Assuming 'image' is the RGB image of the state
        # Ensure the state is correctly shaped as a 2D array (batch size of 1 x state space)
        state = np.reshape(state, [1, state_space])

        # Iterate through steps within the episode
        for time in range(env.max_steps):
            action = agent.act(state)  # Agent selects an action based on the current state
            next_obs, reward, terminated, truncated, info = env.step(action)  # Environment executes the action and returns the next observation and other details
            next_state = preprocess_observation(next_obs)
            # Keep the raw image of the next state for GIF creation
            next_state_img = next_obs['image']

            # Append the current state's image and action to the trajectory
            trajectory.append((env.render(), action))

            # Process the next observation to get the next state
            next_state = np.reshape(next_state, [1, state_space])
            # Determine if the episode has ended, either by termination or truncation
            done = terminated or truncated
            # Store the transition in the agent's memory
            agent.remember(state, action, reward, next_state, done)
            # Update the current state to be the next state
            state = next_state
            state_img = next_state_img  # Update state_img for the next iteration

            # If the episode is done, print the episode stats and break from the loop
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break

            # If there are enough transitions in memory, perform a replay step
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # After each episode, save the recorded trajectory as a GIF
        agent.save_trajectory_as_gif(trajectory, filename=f"trajectory_{e}.gif")
