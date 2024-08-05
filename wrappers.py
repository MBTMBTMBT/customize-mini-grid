import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from sklearn.preprocessing import OneHotEncoder
from gymnasium import spaces


# Initialise separate OneHotEncoders for each feature
object_encoder = OneHotEncoder(categories=[range(len(OBJECT_TO_IDX))], sparse=False)
colour_encoder = OneHotEncoder(categories=[range(len(COLOR_TO_IDX))], sparse=False)
# the state value can either be the state of the doors or direction of the agent.
state_encoder = OneHotEncoder(categories=[range(max(len(STATE_TO_IDX), 4))], sparse=False)
# direction_encoder = OneHotEncoder(categories=[range(4)], sparse=False)

# Fit each encoder to its corresponding range
object_encoder.fit(np.array(range(len(OBJECT_TO_IDX))).reshape(-1, 1))
colour_encoder.fit(np.array(range(len(COLOR_TO_IDX))).reshape(-1, 1))
state_encoder.fit(np.array(range(len(STATE_TO_IDX))).reshape(-1, 1))
# direction_encoder.fit(np.array(range(4)).reshape(-1, 1))


class FullyObsSB3MLPWrapper(FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Define the number of features per grid cell
        num_object_features = len(OBJECT_TO_IDX)  # Number of object types
        num_colour_features = len(COLOR_TO_IDX)  # Number of colours
        num_state_features = max(len(STATE_TO_IDX), 4)  # Number of additional states
        # num_direction_features = 4  # One-hot for direction (4 possible directions)

        # Total number of features for each grid cell
        num_cell_features = num_object_features + num_colour_features + num_state_features

        # Observation space shape after flattening and adding direction
        num_cells = self.env.width * self.env.height
        total_features = num_cells * num_cell_features  # + num_direction_features

        # Define the new observation space
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(total_features,),
            dtype=np.float32,
        )

    def observation(self, obs):
        obs = super().observation(obs)

        # Get the image part of the observation
        image = obs['image']  # (grid_size, grid_size, 3)
        grid_size = image.shape[0]

        # Flatten the image to (grid_size * grid_size, num_channels)
        flattened_image = image.reshape(-1, image.shape[2])

        # One-hot encode object types, colours, and states separately
        object_types_onehot = object_encoder.transform(flattened_image[:, 0].reshape(-1, 1))
        colours_onehot = colour_encoder.transform(flattened_image[:, 1].reshape(-1, 1))
        states_onehot = state_encoder.transform(flattened_image[:, 2].reshape(-1, 1))

        # Concatenate one-hot encodings for the grid cells
        processed_obs = np.concatenate([object_types_onehot, colours_onehot, states_onehot], axis=1)

        # Flatten processed observation
        processed_obs_flat = processed_obs.flatten()

        # Add direction as a separate feature (one-hot encoding)
        # direction_onehot = direction_encoder.transform(np.array([obs['direction']]).reshape(-1, 1)).flatten()

        # Concatenate the flattened grid encoding with the direction encoding
        # final_obs = np.concatenate([processed_obs_flat, direction_onehot])
        final_obs = processed_obs_flat
        # print(final_obs)

        return final_obs


if __name__ == '__main__':
    from custom_env import CustomEnv
    # Initialize the environment and wrapper
    env = CustomEnv(
        txt_file_path='simple_test_corridor.txt',
        display_size=6,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        custom_mission="Find the key and open the door.",
        # render_mode="human"
    )
    env = FullyObsSB3MLPWrapper(env)

    # Reset the environment to see initial observation
    obs, info = env.reset()

    # Print the processed observation shape and content
    print(f"Processed observation shape: {obs.shape}")
    print(f"Processed observation:\n{obs}")

    from minigrid.wrappers import ImgObsWrapper
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(int(2e4))
