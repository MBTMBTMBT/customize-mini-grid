import numpy as np
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from sklearn.preprocessing import OneHotEncoder
from gymnasium import spaces


class FullyObsSB3MLPWrapper(FullyObsWrapper):
    def __init__(self, env, to_print=False, ):
        super().__init__(env)

        self.to_print = to_print

        # Initialise separate OneHotEncoders for each feature
        self.object_encoder = OneHotEncoder(categories=[range(len(OBJECT_TO_IDX))], sparse=False)
        self.colour_encoder = OneHotEncoder(categories=[range(len(COLOR_TO_IDX))], sparse=False)
        # the state value can either be the state of the doors or direction of the agent.
        self.state_encoder = OneHotEncoder(categories=[range(max(len(STATE_TO_IDX), 4))], sparse=False)
        # direction_encoder = OneHotEncoder(categories=[range(4)], sparse=False)

        # Fit each encoder to its corresponding range
        self.object_encoder.fit(np.array(range(len(OBJECT_TO_IDX))).reshape(-1, 1))
        self.colour_encoder.fit(np.array(range(len(COLOR_TO_IDX))).reshape(-1, 1))
        self.state_encoder.fit(np.array(range(len(STATE_TO_IDX))).reshape(-1, 1))
        # direction_encoder.fit(np.array(range(4)).reshape(-1, 1))

        # Define the number of features per grid cell
        self.num_object_features = len(OBJECT_TO_IDX)  # Number of object types
        self.num_colour_features = len(COLOR_TO_IDX)  # Number of colours
        self.num_state_features = max(len(STATE_TO_IDX), 4)  # Number of additional states
        # num_direction_features = 4  # One-hot for direction (4 possible directions)
        self.num_carrying_features = len(OBJECT_TO_IDX)
        self.num_carrying_colour_features = len(COLOR_TO_IDX)
        self.num_carrying_contains_features = len(OBJECT_TO_IDX)
        self.num_carrying_contains_colour_features = len(COLOR_TO_IDX)

        # Total number of features for each grid cell
        self.num_cell_features = self.num_object_features + self.num_colour_features + self.num_state_features

        # Observation space shape after flattening and adding direction
        self.num_cells = self.env.width * self.env.height
        self.total_features = self.num_cells * self.num_cell_features  # + num_direction_features
        self.total_features += self.num_carrying_features + self.num_carrying_colour_features
        self.total_features += self.num_carrying_contains_features + self.num_carrying_contains_colour_features

        # Define the new observation space
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.total_features,),
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
        object_types_onehot = self.object_encoder.transform(flattened_image[:, 0].reshape(-1, 1))
        colours_onehot = self.colour_encoder.transform(flattened_image[:, 1].reshape(-1, 1))
        states_onehot = self.state_encoder.transform(flattened_image[:, 2].reshape(-1, 1))

        # Concatenate one-hot encodings for the grid cells
        processed_obs = np.concatenate([object_types_onehot, colours_onehot, states_onehot], axis=1)

        # Flatten processed observation
        processed_obs_flat = processed_obs.flatten()

        # Add direction as a separate feature (one-hot encoding)
        # direction_onehot = direction_encoder.transform(np.array([obs['direction']]).reshape(-1, 1)).flatten()

        # Add carried things as separate features (one-hot encoding)
        carrying_onehot = self.object_encoder.transform(np.array([obs['carrying']['carrying']]).reshape(-1, 1)).flatten()
        carrying_colour_onehot = self.colour_encoder.transform(np.array([obs['carrying']['carrying_colour']]).reshape(-1, 1)).flatten()
        carrying_contains_onehot = self.object_encoder.transform(np.array([obs['carrying']['carrying_contains']]).reshape(-1, 1)).flatten()
        carrying_contains_colour_onehot = self.colour_encoder.transform(np.array([obs['carrying']['carrying_contains_colour']]).reshape(-1, 1)).flatten()

        # Concatenate the flattened grid encoding with the direction encoding and carried things
        # not needed for direction because it's also in state layer.
        final_obs = np.concatenate([processed_obs_flat, carrying_onehot, carrying_colour_onehot, carrying_contains_onehot, carrying_contains_colour_onehot])
        # final_obs = processed_obs_flat

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


    def decode_to_original_obs(self, one_hot_vector: np.ndarray):
        """
        Decodes a one-hot encoded vector back to the original obs dictionary structure using the class's encoders.
        """
        assert len(one_hot_vector) == self.total_features, "Encoded vector length does not match the expected total features."

        # Calculate where the grid encoding ends and the carried item encoding begins
        grid_encoded_end = self.num_cells * self.num_cell_features
        grid_encoded = one_hot_vector[:grid_encoded_end]
        grid_obs = grid_encoded.reshape(self.env.width * self.env.height, self.num_cell_features)

        # Decoding grid information
        object_types = self.object_encoder.inverse_transform(grid_obs[:, :self.num_object_features]).reshape(self.env.width, self.env.height)
        colours = self.colour_encoder.inverse_transform(grid_obs[:, self.num_object_features:self.num_object_features + self.num_colour_features]).reshape(self.env.width, self.env.height)
        states = self.state_encoder.inverse_transform(grid_obs[:, self.num_object_features + self.num_colour_features:self.num_cell_features]).reshape(self.env.width, self.env.height)

        # Constructing the image from the decoded object types, colours, and states
        image = np.stack([object_types, colours, states], axis=-1)

        # Decode carried item information
        start_idx = grid_encoded_end
        carrying = self.object_encoder.inverse_transform(one_hot_vector[start_idx:start_idx + self.num_object_features].reshape(1, -1))[0]
        start_idx += self.num_object_features
        carrying_colour = self.colour_encoder.inverse_transform(one_hot_vector[start_idx:start_idx + self.num_colour_features].reshape(1, -1))[0]
        start_idx += self.num_colour_features
        carrying_contains = self.object_encoder.inverse_transform(one_hot_vector[start_idx:start_idx + self.num_carrying_contains_features].reshape(1, -1))[0]
        start_idx += self.num_carrying_contains_features
        carrying_contains_colour = self.colour_encoder.inverse_transform(one_hot_vector[start_idx:start_idx + self.num_carrying_contains_colour_features].reshape(1, -1))[0]

        decoded_obs = {
            'image': image,
            'carrying': {
                'carrying': carrying,
                'carrying_colour': carrying_colour,
                'carrying_contains': carrying_contains,
                'carrying_contains_colour': carrying_contains_colour,
            }
        }

        return decoded_obs


def test_encode_decode_consistency(env: FullyObsSB3MLPWrapper, num_epochs=10, num_steps=10):
    """
    Tests the consistency of the encode-decode process by comparing the original observation
    with the decoded observation after encoding, over a number of steps.

    Parameters:
    - env: The environment instance with the FullyObsSB3MLPWrapper.
    - num_epochs: Number of epochs to test the encoding and decoding process.
    - num_steps: Number of steps to perform in each epoch for the test.
    """
    for epoch in range(num_epochs):
        encoded_vector, _ = env.reset()  # Reset the environment to get the initial observation
        broken = False

        # Test the encode-decode consistency for each step
        for step in range(num_steps):
            action = env.action_space.sample()  # Random action
            encoded_vector, _, done, truncated, _ = env.step(action)  # Get new observation after action

            # Check if the episode should end
            if done or truncated:
                print(f"Episode ended at epoch {epoch+1}, step {step+1}: {'due to environment termination.' if done else 'due to truncation.'}")
                break

            # Encode and decode the observation
            decoded_obs = env.decode_to_original_obs(encoded_vector)
            encoded_decoded_vector = env.observation(decoded_obs)

            if not np.array_equal(encoded_vector, encoded_decoded_vector):
                print(f"Test failed at epoch {epoch+1}, step {step+1}: The decoded image does not match the original image.")
                broken = True
                break  # If a test fails, stop further testing

        if broken:
            print("Stopping tests due to failure.")
            break
    else:
        print("All tests passed successfully.")


if __name__ == '__main__':
    from custom_env import CustomEnv
    from minigrid.manual_control import ManualControl

    # Initialize the environment and wrapper
    env = CustomEnv(
        txt_file_path='maps/test.txt',
        display_size=10,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        custom_mission="Find the key and open the door.",
        render_mode="human"
    )
    env = FullyObsSB3MLPWrapper(env, to_print=False)

    test_encode_decode_consistency(env, num_epochs=10, num_steps=10000)

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
