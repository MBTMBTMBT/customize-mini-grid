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
    def __init__(self, env, to_print=False, ):
        super().__init__(env)

        self.to_print = to_print

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
        # not needed because it's also in state layer.
        # final_obs = np.concatenate([processed_obs_flat, direction_onehot])
        final_obs = processed_obs_flat

        if self.to_print:
            # Print the image content and format
            print(f"Image shape: {image.shape}")

            # Print the grid with each layer separately
            # Channel 0: Object types
            print("Object Types (Channel 0):")
            print(image[:, :, 0].transpose(1, 0))

            # Channel 1: Colors
            print("Colors (Channel 1):")
            print(image[:, :, 1].transpose(1, 0))

            # Channel 2: Additional State
            print("Additional State (Channel 2):")
            print(image[:, :, 2].transpose(1, 0))

            direction = obs['direction']
            mission = obs['mission']

            # Print the direction and mission
            print(f"Direction: {direction}")
            print(f"Mission: {mission}")

            print("final obs:")
            print(final_obs)

        return final_obs


if __name__ == '__main__':
    from custom_env import CustomEnv
    from minigrid.manual_control import ManualControl

    # Initialize the environment and wrapper
    env = CustomEnv(
        txt_file_path='maps/door_key.txt',
        display_size=6,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        custom_mission="Find the key and open the door.",
        render_mode="human"
    )
    env = FullyObsSB3MLPWrapper(env, to_print=True)

    manual_control = ManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface

    # # Reset the environment to see initial observation
    # obs, info = env.reset()
    #
    # # Print the processed observation shape and content
    # print(f"Processed observation shape: {obs.shape}")
    # print(f"Processed observation:\n{obs}")
    #
    # from minigrid.wrappers import ImgObsWrapper
    # from stable_baselines3 import PPO
    #
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(int(2e4))
