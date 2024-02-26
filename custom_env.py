from typing import Optional
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import *
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class CustomEnvFromFile(MiniGridEnv):
    def __init__(
            self,
            txt_file_path: str,
            size: int = 10,
            agent_start_pos: tuple[int, int] = (1, 1),
            agent_start_dir: int = 0,
            custom_mission: str = "Explore and interact with objects.",
            max_steps: Optional[int] = None,
            **kwargs,
    ) -> None:
        self.txt_file_path = txt_file_path
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: custom_mission),
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps or 4 * size ** 2,
            **kwargs,
        )
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.mission = custom_mission

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.read_layout_from_file()
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def read_layout_from_file(self) -> None:
        with open(self.txt_file_path, 'r') as file:
            sections = file.read().split('\n\n')
            if len(sections) != 2:
                raise ValueError("File must contain exactly two sections separated by one empty line.")

            layout_lines = sections[0].strip().split('\n')
            color_lines = sections[1].strip().split('\n')

            if len(layout_lines) != len(color_lines) or any(
                    len(layout) != len(color) for layout, color in zip(layout_lines, color_lines)):
                raise ValueError("Object and color matrices must have the same size.")

            for y, (layout_line, color_line) in enumerate(zip(layout_lines, color_lines)):
                for x, (char, color_char) in enumerate(zip(layout_line, color_line)):
                    color = self.char_to_color(color_char)
                    obj = self.char_to_object(char, color)
                    if obj:
                        self.grid.set(x, y, obj)

    def char_to_color(self, char: str) -> str:
        # Expanded map of single characters to color names.
        color_map = {
            'R': 'red',
            'G': 'green',
            'B': 'blue',
            'Y': 'yellow',
            'M': 'magenta',
            'C': 'cyan',
            # Extend with additional colors as needed.
            # Ensure these colors are supported by the objects in MiniGrid.
        }
        return color_map.get(char.upper(), None)

    def char_to_object(self, char: str, color: str) -> Optional[WorldObj]:
        # Assuming `color` is a valid color name string like 'red', 'green', etc.
        obj_map = {
            'W': lambda: Wall(),
            'F': lambda: Floor(),
            'B': lambda: Ball(color),
            'K': lambda: Key(color),
            'X': lambda: Box(color),
            'D': lambda: Door(color, is_locked=True),
            'G': lambda: Goal(),
            'L': lambda: Lava(),
        }
        constructor = obj_map.get(char, None)
        return constructor() if constructor else None


if __name__ == "__main__":
    env = CustomEnvFromFile(
        txt_file_path='simple_test.txt',
        custom_mission="Find the key and open the door.",
        render_mode="human"
    )
    manual_control = ManualControl(env)
    manual_control.start()
