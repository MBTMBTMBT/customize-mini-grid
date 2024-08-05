Custom MiniGrid Environment

This custom MiniGrid environment allows for dynamic creation of grid-based puzzles and scenarios from a simple text file. Each character in the text file represents a different object in the grid, making it easy to design complex environments without altering the codebase.
Environment Setup

The environment is defined in a Python class that extends MiniGridEnv from the MiniGrid package. It reads a .txt file where each character corresponds to a different type of object within the grid. The environment supports a variety of objects, each with unique properties and effects on the agent's behavior.
Objects and Their Representations

    Wall (W): Impassable barriers that the agent cannot move through.
    Floor (F): Empty spaces that the agent can move over.
    Ball (B): Objects that can be picked up by the agent. Represented by the first color in COLOR_NAMES.
    Key (K): Special objects that can be used to unlock doors. Represented by the first color in COLOR_NAMES.
    Box (X): Movable objects that can contain other items. Represented by the first color in COLOR_NAMES.
    Door (D): Barriers that can be opened with a key. Initially locked and represented by the first color in COLOR_NAMES.
    Goal (G): The target area the agent aims to reach. Completing a level usually involves reaching this spot.
    Lava (L): Hazardous areas that terminate the agent's episode upon contact.

Text File Format

The layout of the environment is defined in a .txt file, with each character representing a different object. For example:

    WWWWWWWWWW
    WEEEEEEEEW
    WEEEEEEWDW
    WEEEELLWDW
    WKEEEEKWGW
    WWWWWWWWWW

Color Representation in Grid Layouts

To illustrate the color representation for easier identification of certain objects, consider the following grid layouts 
with Color Representation for Easy Identification

    WWWWWWWWWW
    WEEEEEEEEW
    WEEEEEEWBW
    WEEEELLWRW
    WBEEEERWGW
    WWWWWWWWWW

In this color-coded representation:

    B (Blue): Represents the Ball object for easy identification.
    R (Red): Indicates the Agent's current position and interaction points like keys and doors, making them stand out for easier location.

Key Operations for Manual Control

The environment supports manual control for testing and interaction. Here are the key operations:

    Arrow Keys: Move the agent in the corresponding direction.
    Space: Pick up an object (where applicable).
    Enter: Toggle the door state (open/close) if the agent has a key.

Customization

You can customize the environment by changing the .txt file to create new puzzles or scenarios. Additionally, you can extend the CustomEnvFromFile class to add new object types or behaviors.
Requirements

    Python 3.8
    MiniGrid package
    Gymnasium package

To install dependencies:

    pip install gymnasium minigrid

