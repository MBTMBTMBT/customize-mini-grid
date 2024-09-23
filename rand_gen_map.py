import random


def generate_maze(width, height):
    # Initialize the maze with walls
    maze = [['W' for _ in range(width)] for _ in range(height)]

    # Choose a random starting point that is not on the border
    start_x = random.randint(1, width - 2)
    start_y = random.randint(1, height - 2)
    maze[start_y][start_x] = 'E'

    # Choose a random goal position different from the starting position
    goal_x, goal_y = start_x, start_y
    counter = 0
    while goal_x == start_x and goal_y == start_y:
        goal_x = random.randint(1, width - 2)
        goal_y = random.randint(1, height - 2)
        counter += 1
        if counter > 512:
            raise RuntimeError("Generation fails.")
    maze[goal_y][goal_x] = 'G'

    # Create a path from start to goal using Depth First Search (DFS)
    def carve_path(x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == 'W':
                # Count the number of adjacent non-walls to prevent loops
                adjacent_non_walls = sum(1 for dx2, dy2 in directions
                                         if 0 <= nx + dx2 < width and 0 <= ny + dy2 < height and maze[ny + dy2][
                                             nx + dx2] in ('E', 'G'))
                if adjacent_non_walls < 2:
                    maze[ny][nx] = 'E'
                    carve_path(nx, ny)

    carve_path(start_x, start_y)

    # Ensure there is a path from the start to the goal by backtracking from the goal
    def ensure_path(x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width - 1 and 1 <= ny < height - 1 and maze[ny][nx] == 'W':
                # Set the cell as an empty path to ensure connectivity
                maze[ny][nx] = 'E'
                ensure_path(nx, ny)

    ensure_path(goal_x, goal_y)

    # Convert maze to the string format
    return '\n'.join(''.join(row) for row in maze)


if __name__ == '__main__':
    # Example usage
    width = 8
    height = 8
    maze = generate_maze(width, height)
    print(maze)
