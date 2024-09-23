import random
from typing import Optional, Tuple, List, Dict, Any, SupportsFloat
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.core.world_object import WorldObj
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX


class CustomEnv(MiniGridEnv):
    """
    A custom MiniGrid environment that loads its layout and object properties from a text file.

    Attributes:
        txt_file_path (str): Path to the text file containing the environment layout.
        layout_size (int): The size of the environment, either specified or determined from the file.
        agent_start_pos (tuple[int, int]): Starting position of the agent.
        agent_start_dir (int): Initial direction the agent is facing.
        mission (str): Custom mission description.
    """

    def __init__(
            self,
            txt_file_path: Optional[str],
            rand_gen_shape: Tuple[int, int],
            display_size: Optional[int] = None,
            display_mode: Optional[str] = "middle",
            random_rotate: bool = False,
            random_flip: bool = False,
            agent_start_pos: Tuple[int, int] or None = None,
            agent_start_dir: Optional[int] = None,
            custom_mission: str = "Explore and interact with objects.",
            max_steps: Optional[int] = 100000,
            **kwargs,
    ) -> None:
        """
        Initializes the custom environment.

        If 'size' is not specified, it determines the size based on the content of the given text file.
        """
        self.txt_file_path = txt_file_path
        self.rand_gen_shape = rand_gen_shape

        assert txt_file_path is not None or rand_gen_shape is not None, "Either 'txt_file_path' or 'rand_gen_shape' must be specified."
        assert not (txt_file_path is not None and rand_gen_shape is not None), "Only one of 'txt_file_path' and 'rand_gen_shape' can be specified."

        # Determine the size of the environment from given file
        self.layout_size = self.determine_layout_size()
        if display_size is None:
            self.display_size = self.layout_size
        else:
            self.display_size = display_size

        # assert: display mode is either middle or random
        assert display_mode in ["middle", "random"]
        self.display_mode = display_mode

        self.random_rotate = random_rotate
        self.random_flip = random_flip

        # assert: the expected display size should be larger than or same as actual size
        assert self.display_size >= self.layout_size

        if agent_start_dir is None:
            agent_start_dir = random.choice(list(range(0, 5)))

        self.random_layout = False
        if self.txt_file_path:
            self.layout, self.colour_layout = self.read_file()
        else:
            self.random_layout = True
            self.layout, self.colour_layout = self.generate_random_maze()

        # Initialize the MiniGrid environment with the determined size
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=self.display_size,  # here should be the actual shown size
            max_steps=max_steps,
            **kwargs,
        )
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.mission = custom_mission

        self.step_count = 0
        self.skip_reset = False

    def determine_layout_size(self) -> int:
        if self.txt_file_path:
            with open(self.txt_file_path, 'r') as file:
                sections = file.read().split('\n\n')
                layout_lines = sections[0].strip().split('\n')
                height = len(layout_lines)
                width = max(len(line) for line in layout_lines)
                return max(width, height)
        else:
            return max(self.rand_gen_shape)

    def read_file(self) -> Tuple[List[List[Optional[WorldObj]]], List[List[Optional[str]]]]:
        layout = []
        colour_layout = []
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            if len(sections) != 2:
                raise ValueError("File must contain exactly two sections separated by one empty line.")

            layout_lines, color_lines = sections[0].strip().split('\n'), sections[1].strip().split('\n')

            if len(layout_lines) != len(color_lines) or any(
                    len(layout) != len(color) for layout, color in zip(layout_lines, color_lines)):
                raise ValueError("Object and colour matrices must have the same size.")

            for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
                line = []
                colour_line = []
                for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                    line.append(char)
                    colour_line.append(color_char)
                layout.append(line)
                colour_layout.append(colour_line)
        return layout, colour_layout

    def generate_random_maze(self) -> Tuple[List[List[str]], List[List[str]]]:
        width, height = self.rand_gen_shape

        # Initialize the maze with walls
        maze = [['W' for _ in range(width)] for _ in range(height)]

        # Choose a random starting point that is not on the border
        start_x = random.randint(1, width - 2)
        start_y = random.randint(1, height - 2)
        maze[start_y][start_x] = 'E'

        # Choose a random goal position different from the starting position
        goal_x, goal_y = start_x, start_y
        while goal_x == start_x and goal_y == start_y:
            goal_x = random.randint(1, width - 2)
            goal_y = random.randint(1, height - 2)
        maze[goal_y][goal_x] = 'G'

        # Create a path from start to goal using DFS
        def carve_path(x, y):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == 'W':
                    adjacent_non_walls = sum(1 for dx2, dy2 in directions
                                             if 0 <= nx + dx2 < width and 0 <= ny + dy2 < height and maze[ny + dy2][
                                                 nx + dx2] in ('E', 'G'))
                    if adjacent_non_walls < 2:
                        maze[ny][nx] = 'E'
                        carve_path(nx, ny)

        carve_path(start_x, start_y)

        # Ensure path from goal is connected
        def ensure_path(x, y):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == 'W':
                    maze[ny][nx] = 'E'
                    ensure_path(nx, ny)

        ensure_path(goal_x, goal_y)

        # Duplicate maze as colour layout
        color_maze = [row[:] for row in maze]

        return maze, color_maze

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generates the grid for the environment based on the layout specified in the text file.
        """
        self.grid = Grid(width, height)
        for x in range(width):
            for y in range(height):
                self.grid.set(x, y, Wall())
        free_width = self.display_size - len(self.layout[0])
        free_height = self.display_size - len(self.layout)
        if self.display_mode == "middle":
            anchor_x = free_width // 2
            anchor_y = free_height // 2
        elif self.display_mode == "random":
            if free_width > 0:
                anchor_x = random.choice(list(range(free_width)))
            else:
                anchor_x = 0
            if free_height > 0:
                anchor_y = random.choice(list(range(free_height)))
            else:
                anchor_y = 0
        else:
            raise ValueError("Invalid display mode.")
        if self.random_rotate:
            image_direction = random.choice([0, 1, 2, 3])
        else:
            image_direction = 0
        if self.random_flip:
            flip = random.choice([0, 1])
        else:
            flip = 0

        self.empty_list = []
        for y, (obj_line, colour_line) in enumerate(zip(self.layout, self.colour_layout)):
            for x, (char, color_char) in enumerate(zip(obj_line, colour_line)):
                colour = self.char_to_colour(color_char)
                obj = self.char_to_object(char, colour)
                if obj is not None:
                    obj.cur_pos = (x, y)  # to set the correct position og the obj
                x_coord, y_coord = anchor_x + x, anchor_y + y
                x_coord, y_coord = rotate_coordinate(x_coord, y_coord, image_direction, self.display_size)
                x_coord, y_coord = flip_coordinate(x_coord, y_coord, flip, self.display_size)
                self.grid.set(x_coord, y_coord, obj)
                if obj is None:
                    self.empty_list.append((x_coord, y_coord))

        # update agent's coordinate
        if self.agent_start_pos is not None:
            x_coord, y_coord = anchor_x + self.agent_start_pos[0], anchor_y + self.agent_start_pos[1]
            x_coord, y_coord = rotate_coordinate(x_coord, y_coord, image_direction, self.display_size)
            x_coord, y_coord = flip_coordinate(x_coord, y_coord, flip, self.display_size)
            self.agent_pos = (x_coord, y_coord)
        else:
            self.agent_pos = random.choice(self.empty_list)

        self.agent_dir = flip_direction(rotate_direction(self.agent_start_dir, image_direction), flip)

    def reset(
        self,
        *,
        seed: int or None = None,
        options: Dict[str, Any] or None = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if not self.skip_reset:
            super().reset(seed=seed)

            # Reinitialize episode-specific variables
            self.agent_pos = (-1, -1)
            self.agent_dir = -1

            if self.random_layout:
                self.layout, self.colour_layout = self.generate_random_maze()

            # Generate a new random grid at the start of each episode
            self._gen_grid(self.width, self.height)

            # These fields should be defined by _gen_grid
            assert (
                self.agent_pos >= (0, 0)
                if isinstance(self.agent_pos, tuple)
                else all(self.agent_pos >= 0) and self.agent_dir >= 0
            )

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos)
            assert start_cell is None or start_cell.can_overlap()

            # Item picked up, being carried, initially nothing
            self.carrying = None

            # Step count since episode start
            self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        obs["carrying"] = {
            "carrying": 0,
            "carrying_colour": 0,
            # "carrying_contains": 0,
            # "carrying_contains_colour": 0,
        }

        if self.carrying is not None:
            carrying = OBJECT_TO_IDX[self.carrying.type]
            carrying_colour = COLOR_TO_IDX[self.carrying.color]

            obs["carrying"] = {
                "carrying": carrying,
                "carrying_colour": carrying_colour,
                # "carrying_contains": 0,
                # "carrying_contains_colour": 0,
            }

        obs["overlap"] = {
            "obj": 0,
            "colour": 0,
        }

        overlap = self.grid.get(*self.agent_pos)
        if overlap is not None:
            overlap_colour = COLOR_TO_IDX[overlap.color]
            obs["overlap"] = {
                "obj": OBJECT_TO_IDX[overlap.type],
                "colour": overlap_colour,
            }

        return obs, {}

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        reward = -0.05  # give negative reward for normal steps

        carrying = 0  # object carried by the agent
        carrying_colour = 0
        carrying_contains = 0
        carrying_contains_colour = 0  # assume a box cannot contain another box.

        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = 1  # give settled 1 as reward,
                # instead of the original 1 - 0.9 * (self.step_count / self.max_steps)
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None or self.carrying == 0:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        obs["carrying"] = {
            "carrying": 0,
            "carrying_colour": 0,
            # "carrying_contains": carrying_contains,
            # "carrying_contains_colour": carrying_contains_colour,
        }

        if self.carrying is not None and self.carrying != 0:
            carrying = OBJECT_TO_IDX[self.carrying.type]
            carrying_colour = COLOR_TO_IDX[self.carrying.color]
            # carrying_contains = 0 if self.carrying.contains is None else OBJECT_TO_IDX[self.carrying.contains.type]
            # carrying_contains_colour = 0 if self.carrying.contains is None else COLOR_TO_IDX[self.carrying.contains.color]

            obs["carrying"] = {
                "carrying": carrying,
                "carrying_colour": carrying_colour,
                # "carrying_contains": carrying_contains,
                # "carrying_contains_colour": carrying_contains_colour,
            }

        obs["overlap"] = {
            "obj": 0,
            "colour": 0,
        }

        overlap = self.grid.get(*self.agent_pos)
        if overlap is not None:
            overlap_colour = COLOR_TO_IDX[overlap.color]
            obs["overlap"] = {
                "obj": OBJECT_TO_IDX[overlap.type],
                "colour": overlap_colour,
            }

        return obs, reward, terminated, truncated, {}

    def set_env_by_obs(self, obs: ObsType):
        """
        NOTES: setting the environment this way, Box will always be empty!!!
        """
        # self.skip_reset = True
        # values needed:
        # self.agent_pos, self.agent_dir
        # self.grid needs to be reset
        # self.carrying, and everything within this carried object
        image = obs["image"]
        object_channel = image[:, :, 0]
        indices = np.argwhere(object_channel == OBJECT_TO_IDX["agent"])
        assert len(indices) == 1, "Only one agent can be in the map."
        self.agent_pos = tuple(indices[0])
        self.agent_dir = image[:, :, 2][self.agent_pos]
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                obj = self.int_to_object(int(image[x, y, 0]), IDX_TO_COLOR[image[x, y, 1]])
                if obj is not None and obj.type == "door":
                    obj.is_open = image[x, y, 2] == STATE_TO_IDX["open"]
                    obj.is_locked = image[x, y, 2] == STATE_TO_IDX["locked"]
                self.grid.set(x, y, obj)
        if obs["overlap"]["obj"] is not None:
            obj = self.int_to_object(obs["overlap"]["obj"][0], IDX_TO_COLOR[obs["overlap"]["colour"][0]])
            if obj is not None and obj.type == "door":
                obj.is_open = True  # overlap - for sure it's open
            self.grid.set(self.agent_pos[0], self.agent_pos[1], obj)
        self.carrying = self.int_to_object(obs['carrying']['carrying'][0], IDX_TO_COLOR[obs['carrying']['carrying_colour'][0]])
        if self.carrying is not None:
            self.carrying.cur_pos = np.array([-1, -1])
        self.skip_reset = True
        return self.reset()

    def char_to_colour(self, char: str) -> Optional[str]:
        """
        Maps a single character to a color name supported by MiniGrid objects.

        Args:
            char (str): A character representing a color.

        Returns:
            Optional[str]: The name of the color, or None if the character is not recognized.
        """
        color_map = {'R': 'red', 'G': 'green', 'B': 'blue', 'P': 'purple', 'Y': 'yellow', 'G': 'grey'}
        return color_map.get(char.upper(), None)

    def char_to_object(self, char: str, color: str) -> Optional[WorldObj]:
        """
        Maps a character (and its associated color) to a MiniGrid object.

        Args:
            char (str): A character representing an object type.
            color (str): The color of the object.

        Returns:
            Optional[WorldObj]: The MiniGrid object corresponding to the character and color, or None if unrecognized.
        """
        obj_map = {
            'W': lambda: Wall(), 'F': lambda: Floor(), 'B': lambda: Ball(color),
            'K': lambda: Key(color), 'X': lambda: Box(color), 'D': lambda: Door(color, is_locked=True),
            'G': lambda: Goal(), 'L': lambda: Lava(),
        }
        constructor = obj_map.get(char, None)
        return constructor() if constructor else None

    def int_to_object(self, val: int, color: str) -> Optional[WorldObj]:
        obj_str = IDX_TO_OBJECT[val]
        obj_map = {
            'wall': lambda: Wall(), 'floor': lambda: Floor(), 'ball': lambda: Ball(color),
            'key': lambda: Key(color), 'box': lambda: Box(color), 'door': lambda: Door(color, is_locked=True),
            'goal': lambda: Goal(), 'lava': lambda: Lava(),
        }
        constructor = obj_map.get(obj_str, None)
        return constructor() if constructor else None


def rotate_coordinate(x, y, rotation_mode, n):
    """
    Rotate a 2D coordinate in a gridworld.

    Parameters:
    x, y (int): Original coordinates.
    rotation_mode (int): Rotation mode (0, 1, 2, 3).
    n (int): Dimension of the matrix.

    Returns:
    tuple: The new coordinates (new_x, new_y) after rotation.
    """
    if rotation_mode == 0:
        # No rotation
        return x, y
    elif rotation_mode == 1:
        # Clockwise rotation by 90 degrees
        return y, n - 1 - x
    elif rotation_mode == 2:
        # Clockwise rotation by 180 degrees
        return n - 1 - x, n - 1 - y
    elif rotation_mode == 3:
        # Clockwise rotation by 270 degrees
        return n - 1 - y, x
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_coordinate(x, y, flip_mode, n):
    """
    Flip a 2D coordinate in a gridworld along the x-axis.

    Parameters:
    x, y (int): Original coordinates.
    flip_mode (int): Flip mode (0 for no flip, 1 for flip along x-axis).
    n (int): Dimension of the matrix.

    Returns:
    tuple: The new coordinates (new_x, new_y) after flipping.
    """
    if flip_mode == 0:
        # No flip
        return x, y
    elif flip_mode == 1:
        # Flip along the x-axis
        return n - 1 - x, y
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


def rotate_direction(direction, rotation_mode):
    """
    Rotate a direction in a gridworld.

    Parameters:
    direction (int): Original direction (0, 1, 2, 3).
    rotation_mode (int): Rotation mode (0, 1, 2, 3).

    Returns:
    int: New direction after rotation.
    """
    # Apply rotation by adding the rotation mode and taking modulo 4 to cycle back to 0 after 3.
    if rotation_mode in [0, 1, 2, 3]:
        return (direction + rotation_mode) % 4
    else:
        raise ValueError("Invalid rotation mode. Please choose between 0, 1, 2, or 3.")


def flip_direction(direction, flip_mode):
    """
    Flip a direction in a gridworld along the x-axis.

    Parameters:
    direction (int): Original direction (0, 1, 2, 3).
    flip_mode (int): Flip mode (0 for no flip, 1 for flip).

    Returns:
    int: New direction after flipping.
    """
    if flip_mode == 0:
        # No flip
        return direction
    elif flip_mode == 1:
        # Flip direction: right/left remain the same, up/down are flipped
        flip_map = {0: 0, 1: 3, 2: 2, 3: 1}
        return flip_map[direction]
    else:
        raise ValueError("Invalid flip mode. Please choose between 0 and 1.")


if __name__ == "__main__":
    env = CustomEnv(
        txt_file_path=None,
        rand_gen_shape=(10, 10),
        display_size=20,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        custom_mission="Find the key and open the door.",
        render_mode="human",
    )
    manual_control = ManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
