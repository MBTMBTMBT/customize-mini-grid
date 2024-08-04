import random
from typing import Optional, Tuple, List
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.core.world_object import WorldObj
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


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
            txt_file_path: str,
            display_size: Optional[int] = None,
            display_mode: Optional[str] = "middle",
            random_rotate: bool = False,
            random_flip: bool = False,
            agent_start_pos: Tuple[int, int] = (1, 1),
            agent_start_dir: Optional[int] = None,
            custom_mission: str = "Explore and interact with objects.",
            max_steps: Optional[int] = 16384,
            **kwargs,
    ) -> None:
        """
        Initializes the custom environment.

        If 'size' is not specified, it determines the size based on the content of the given text file.
        """
        self.txt_file_path = txt_file_path

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

        self.layout, self.colour_layout = self.read_file()

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

    def determine_layout_size(self) -> int:
        """
        Reads the layout from the file to determine the environment's size based on its width and height.

        Returns:
            int: The larger value between the height and width of the layout to ensure a square grid.
        """
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            layout_lines = sections[0].strip().split('\n')
            height = len(layout_lines)
            width = max(len(line) for line in layout_lines)
            return max(width, height)

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
        for y, (obj_line, colour_line) in enumerate(zip(self.layout, self.colour_layout)):
            for x, (char, color_char) in enumerate(zip(obj_line, colour_line)):
                colour = self.char_to_colour(color_char)
                obj = self.char_to_object(char, colour)
                x_coord, y_coord = anchor_x + x, anchor_y + y
                x_coord, y_coord = rotate_coordinate(x_coord, y_coord, image_direction, self.display_size)
                x_coord, y_coord = flip_coordinate(x_coord, y_coord, flip, self.display_size)
                self.grid.set(x_coord, y_coord, obj)

        # update agent's coordinate
        x_coord, y_coord = anchor_x + self.agent_start_pos[0], anchor_y + self.agent_start_pos[1]
        x_coord, y_coord = rotate_coordinate(x_coord, y_coord, image_direction, self.display_size)
        x_coord, y_coord = flip_coordinate(x_coord, y_coord, flip, self.display_size)
        self.agent_pos = (x_coord, y_coord)
        self.agent_dir = flip_direction(rotate_direction(self.agent_start_dir, image_direction), flip)

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


class CustomEnvFromFile(MiniGridEnv):
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
            txt_file_path: str,
            size: Optional[int] = None,
            agent_start_pos: Tuple[int, int] = (1, 1),
            agent_start_dir: int = 2,
            custom_mission: str = "Explore and interact with objects.",
            max_steps: Optional[int] = None,
            **kwargs,
    ) -> None:
        """
        Initializes the custom environment.

        If 'size' is not specified, it determines the size based on the content of the given text file.
        """
        self.txt_file_path = txt_file_path
        # Determine the size of the environment if not provided
        self.layout_size = self.determine_layout_size() if size is None else size
        # Initialize the MiniGrid environment with the determined size
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=self.layout_size,
            see_through_walls=False,
            max_steps=max_steps or 4 * self.layout_size ** 2,
            **kwargs,
        )
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.mission = custom_mission

    def determine_layout_size(self) -> int:
        """
        Reads the layout from the file to determine the environment's size based on its width and height.

        Returns:
            int: The larger value between the height and width of the layout to ensure a square grid.
        """
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            layout_lines = sections[0].strip().split('\n')
            height = len(layout_lines)
            width = max(len(line) for line in layout_lines)
            return max(width, height)

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Generates the grid for the environment based on the layout specified in the text file.
        """
        self.grid = Grid(width, height)
        self.read_layout_from_file()
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def read_layout_from_file(self) -> None:
        """
        Parses the text file specified by 'txt_file_path' to set the objects in the environment's grid.
        """
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            if len(sections) != 2:
                raise ValueError("File must contain exactly two sections separated by one empty line.")

            layout_lines, color_lines = sections[0].strip().split('\n'), sections[1].strip().split('\n')

            if len(layout_lines) != len(color_lines) or any(
                    len(layout) != len(color) for layout, color in zip(layout_lines, color_lines)):
                raise ValueError("Object and color matrices must have the same size.")

            for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
                for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                    color = self.char_to_color(color_char)
                    obj = self.char_to_object(char, color)
                    if obj:
                        self.grid.set(x, y, obj)  # Place the object on the grid

    def char_to_color(self, char: str) -> Optional[str]:
        """
        Maps a single character to a color name supported by MiniGrid objects.

        Args:
            char (str): A character representing a color.

        Returns:
            Optional[str]: The name of the color, or None if the character is not recognized.
        """
        color_map = {'R': 'red', 'G': 'green', 'B': 'blue', 'Y': 'yellow', 'M': 'magenta', 'C': 'cyan'}
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


if __name__ == "__main__":
    # Example usage of the CustomEnvFromFile class
    env = CustomEnv(
        txt_file_path='test.txt',
        display_size=20,
        display_mode="random",
        random_rotate=True,
        random_flip=True,
        custom_mission="Find the key and open the door.",
        render_mode="human"
    )
    manual_control = ManualControl(env)  # Allows manual control for testing and visualization
    manual_control.start()  # Start the manual control interface
