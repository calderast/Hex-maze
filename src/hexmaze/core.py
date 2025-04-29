"""
core.py

This module contains core functions for hex maze analysis, 
including generating hex maze configurations and calculating various hex maze attributes.
"""

import networkx as nx
import numpy as np
import math
from collections import Counter

from .utils import (
    maze_to_graph, 
    maze_to_barrier_set, 
    create_empty_hex_maze
)
from .barrier_shift import get_barrier_changes

# This is defined up here because we use it to set up constants
def get_subpaths(path, length):
    """Helper. Given a path, return a set of all sub-paths of the specified length."""
    return {tuple(path[i : i + length]) for i in range(len(path) - length + 1)}

################################# Set up all of our constants #################################

# Barriers can exist in any open hex, excluding the hexes right next to the reward ports
POSSIBLE_BARRIERS = np.arange(5, 48)
POSSIBLE_BARRIERS = POSSIBLE_BARRIERS[~np.isin(POSSIBLE_BARRIERS, [5, 6, 33, 38, 43, 47])]

# Minimum distance from port to critical choice point = 6 (including port hex)
ILLEGAL_CHOICE_POINTS_6 = [4, 6, 5, 11, 8, 10, 7, 9, 49, 38, 47, 32, 42, 27, 37, 46, 48, 43, 33, 39, 28, 44, 34, 23]

# Max straight path length to reward port = 6 hexes. (illegal paths are 7+)
MAX_STRAIGHT_PATH_TO_PORT = 6
STRAIGHT_PATHS_TO_PORTS = [
    [1, 4, 6, 8, 11, 14, 18, 22, 27, 32, 38, 49, 2],
    [1, 4, 5, 7, 9, 12, 15, 19, 23, 28, 33, 48, 3],
    [2, 49, 47, 42, 46, 41, 45, 40, 44, 39, 43, 48, 3],
]
# Max straight path length inside maze = 6 hexes. (illegal paths are 7+)
MAX_STRAIGHT_PATH_INSIDE_MAZE = 6
STRAIGHT_PATHS_INSIDE_MAZE = [
    [5, 7, 10, 13, 17, 21, 26, 31, 37, 42, 47],
    [9, 12, 16, 20, 25, 30, 36, 41, 46],
    [6, 8, 10, 13, 16, 20, 24, 29, 34, 39, 43],
    [11, 14, 17, 21, 25, 30, 35, 40, 44],
    [38, 32, 37, 31, 36, 30, 35, 29, 34, 28, 33],
    [27, 22, 26, 21, 25, 20, 24, 19, 23],
]
# For training mazes, the max straight path length = 8 hexes (illegal paths are 9+)
MAX_STRAIGHT_PATH_TRAINING = 8

# Get all illegal straight paths to ports
illegal_straight_paths_list = []
for path in STRAIGHT_PATHS_TO_PORTS:
    for sub_path in get_subpaths(path, MAX_STRAIGHT_PATH_TO_PORT + 1):
        illegal_straight_paths_list.append(sub_path)

# Store illegal straight paths as a set of tuples for O(1) lookup time
ILLEGAL_STRAIGHT_PATHS_TO_PORT = {tuple(path) for path in illegal_straight_paths_list}

# Get all illegal straight paths inside the maze
illegal_straight_paths_list = []
for path in STRAIGHT_PATHS_INSIDE_MAZE:
    for sub_path in get_subpaths(path, MAX_STRAIGHT_PATH_INSIDE_MAZE + 1):
        illegal_straight_paths_list.append(sub_path)

# Store illegal straight paths as a set of tuples for O(1) lookup time
ILLEGAL_STRAIGHT_PATHS_INSIDE_MAZE = {tuple(path) for path in illegal_straight_paths_list}

# Get all illegal straight paths for training mazes
illegal_straight_paths_list_training = []
for path in STRAIGHT_PATHS_TO_PORTS:
    for sub_path in get_subpaths(path, MAX_STRAIGHT_PATH_TRAINING + 1):
        illegal_straight_paths_list_training.append(sub_path)
for path in STRAIGHT_PATHS_INSIDE_MAZE:
    for sub_path in get_subpaths(path, MAX_STRAIGHT_PATH_TRAINING + 1):
        illegal_straight_paths_list_training.append(sub_path)

# Store illegal straight paths as a set of tuples for O(1) lookup time
ILLEGAL_STRAIGHT_PATHS_TRAINING = {tuple(path) for path in illegal_straight_paths_list_training}


# Define the public interface for this module
__all__ = [
    "get_critical_choice_points", 
    "get_all_choice_points",
    "get_optimal_paths_between_ports",
    "get_optimal_paths",
    "get_reward_path_lengths",
    "get_path_independent_hexes_to_port",
    "get_hexes_from_port",
    "get_hexes_within_distance",
    "distance_to_nearest_hex_in_group",
    "get_hexes_on_optimal_paths",
    "get_non_dead_end_hexes",
    "get_dead_end_hexes",
    "get_non_optimal_non_dead_end_hexes",
    "classify_maze_hexes",
    "get_dead_ends",
    "get_dead_end_lengths",
    "get_num_dead_ends",
    "is_valid_path",
    "divide_into_thirds",
    "get_choice_direction",
    "has_illegal_straight_path",
    "is_valid_maze",
    "is_valid_training_maze",
    "generate_good_maze",
    "rotate_hex",
    "reflect_hex",
    "get_rotated_barriers",
    "get_reflected_barriers",
    "get_isomorphic_mazes",
    "get_maze_attributes",
    "get_barrier_sequence_attributes",
]


def get_critical_choice_points(maze) -> set[int]:
    """
    Given a hex maze, find all critical choice points between reward ports 1, 2, and 3.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[int]: Set of ints representing hexes that are the critical choice points for this maze
    """

    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    paths12 = list(nx.all_shortest_paths(graph, source=1, target=2))
    paths13 = list(nx.all_shortest_paths(graph, source=1, target=3))
    paths23 = list(nx.all_shortest_paths(graph, source=2, target=3))

    choice_points = set()
    # all choice points from port 1
    for path_a in paths12:
        for path_b in paths13:
            shared_path = [hex for hex in path_a if hex in path_b]
            choice_points.add(shared_path[-1])

    # all choice points from port 2
    for path_a in paths12:
        for path_b in paths23:
            shared_path = [hex for hex in path_a[::-1] if hex in path_b]
            choice_points.add(shared_path[-1])

    # all choice points from port 3
    for path_a in paths13:
        for path_b in paths23:
            shared_path = [hex for hex in path_a[::-1] if hex in path_b[::-1]]
            choice_points.add(shared_path[-1])
    return choice_points


def get_all_choice_points(maze) -> set[int]:
    """
    Given a hex maze, find all potential choice points (hexes connected to 3 other
    hexes, where a rat coming from a neighboring hex faces a left/right choice
    of 2 other neighboring hexes)

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[int]: Set of ints representing all hexes that are choice points for this maze
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Choice hexes are all hexes with exactly 3 neighbors
    choice_hexes = {hex for hex, degree in graph.degree() if degree == 3}
    return choice_hexes


def get_optimal_paths_between_ports(maze) -> list[list]:
    """
    Given a hex maze, return a list of all optimal paths between reward ports in the maze.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        list[list]: A list of lists representing all optimal paths (in hexes)
            from reward port 1 to 2, 1 to 3, and 2 to 3
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    optimal_paths = []
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=1, target=2)))
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=1, target=3)))
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=2, target=3)))
    return optimal_paths


def get_optimal_paths(maze, start_hex: int, target_hex: int) -> list[list]:
    """
    Given a hex maze, return a list of all optimal paths
    from the start_hex to the target_hex in the maze.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        start_hex (int): The starting hex in the maze
        target_hex (int): The target hex in the maze

    Returns:
        list[list]: A list of lists representing all optimal path(s) (in hexes)
            from the start hex to the target hex
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    return list(nx.all_shortest_paths(graph, source=start_hex, target=target_hex))


def get_reward_path_lengths(maze) -> list:
    """
    Given a hex maze, get the minimum path lengths (in hexes) between reward ports 1, 2, and 3.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        list: Reward path lengths in form [length12, length13, length23]
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Get length of optimal paths between reward ports
    len12 = nx.shortest_path_length(graph, source=1, target=2) + 1
    len13 = nx.shortest_path_length(graph, source=1, target=3) + 1
    len23 = nx.shortest_path_length(graph, source=2, target=3) + 1

    return [len12, len13, len23]


def get_path_independent_hexes_to_port(maze, reward_port) -> set[int]:
    """
    Find all path-independent hexes to a reward port, defined as hexes
    that a rat MUST run through to get to the port regardless of which
    path he is taking/his reward port of origin. These are the same as
    the hexes the rat must run through when leaving this port before he
    reaches the (first) critical choice point.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        reward_port (int or str): The reward port (1, 2, 3, or A, B, C)

    Returns:
        set[int]: The path-independent hexes the rat must always run through
            when going to and from this reward port
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Create a mapping so we can handle 1, 2, 3 or A, B, C to specify reward ports
    port_hex_map = {"A": 1, "B": 2, "C": 3, 1: 1, 2: 2, 3: 3}
    port_hex = port_hex_map[reward_port]

    # Get all shortest paths between reward_port and the other 2 ports
    other_ports = [1, 2, 3]
    other_ports.remove(port_hex)
    paths_a = list(nx.all_shortest_paths(graph, source=port_hex, target=other_ports[0]))
    paths_b = list(nx.all_shortest_paths(graph, source=port_hex, target=other_ports[1]))

    # The path-independent hexes are the common hexes on the shortest
    # paths between the reward port and both other ports
    path_independent_hexes = set()
    for path_a in paths_a:
        for path_b in paths_b:
            shared_path = [hex for hex in path_a if hex in path_b]
            path_independent_hexes.update(shared_path)

    return path_independent_hexes


def get_hexes_from_port(maze, start_hex: int, reward_port) -> int:
    """
    Find the minimum number of hexes from a given hex to a
    chosen reward port for a given maze configuration.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        start_hex (int): The hex to calculate distance from
        reward_port (int or str): The reward port (1, 2, 3, or A, B, C)

    Returns:
        int: The number of hexes from start_hex to reward_port
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Create a mapping so we can handle 1, 2, 3 or A, B, C to specify reward ports
    port_hex_map = {"A": 1, "B": 2, "C": 3, 1: 1, 2: 2, 3: 3}
    port_hex = port_hex_map[reward_port]

    # Get the shortest path length between start_hex and the reward port
    return nx.shortest_path_length(graph, source=start_hex, target=port_hex)


def get_hexes_within_distance(maze, start_hex: int, max_distance=math.inf, min_distance=1) -> set[int]:
    """
    Find all hexes within a certain hex distance from the start_hex (inclusive).
    Hexes directly adjacent to the start_hex are considered 1 hex away,
    hexes adjacent to those are 2 hexes away, etc.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        start_hex (int): The hex to calculate distance from
        max_distance (int): Maximum distance in hexes from the start hex (inclusive)
        min_distance (int): Minimum distance in hexes from the start hex (inclusive).
            Defaults to 1 to not include the start_hex

    Returns:
        set[int]: Set of hexes in the maze that are within the specified distance from the start_hex
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Get a dict of shortest path lengths from the start_hex to all other hexes
    shortest_paths = nx.single_source_shortest_path_length(graph, start_hex)

    # Get hexes that are between min_distance and max_distance (inclusive)
    hexes_within_distance = {hex for hex, dist in shortest_paths.items() if min_distance <= dist <= max_distance}
    return hexes_within_distance


def distance_to_nearest_hex_in_group(maze, hexes, target_group) -> int | dict:
    """
    Calculate the distance (in hexes) between each hexes in 'hexes' and
    the nearest hex in the target group. Often used to calculate how far into
    a dead end a given hex is. Also useful for calculating a hex's distance
    from the optimal paths.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        hexes (int | list | set): Hex(es) to calculate distance for
        target_group (set): The group of hexes to compute distance to

    Returns:
        int | dict: Shortest distance to any hex in target_group
            (int if a single hex was passed, or a dict for multiple hexes)
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Make hexes iterable so this works with a single hex too
    single_hex = not isinstance(hexes, (list, set, tuple))
    hexes = [hexes] if single_hex else hexes

    distances = {}
    for hex in hexes:
        # Find the shortest path lengths from this hex to all hexes in the maze
        path_lengths = nx.single_source_shortest_path_length(graph, hex)
        # Filter for only paths from the hex to hexes in the target_group
        valid_lengths = [dist for target_hex, dist in path_lengths.items() if target_hex in target_group]
        # Get the min distance from this hex to any of the hexes in the target group
        distances[hex] = min(valid_lengths, default=float('inf'))
    # Return an int if only one hex, or a dict if multiple
    return next(iter(distances.values())) if single_hex else distances


def get_hexes_on_optimal_paths(maze) -> set[int]:
    """
    Given a hex maze, return a set of all hexes on 
    optimal paths between reward ports.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[int]: A set of hexes appearing on any optimal path
            between reward ports
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    hexes_on_optimal_paths = set()
    for source_hex, target_hex in [(1, 2), (1, 3), (2, 3)]:
        # Get the shortest path(s) between this pair of reward ports
        optimal_paths_between_ports = nx.all_shortest_paths(graph, source=source_hex, target=target_hex)
        # For each path, add all hexes on the path to our set of hexes on optimal paths
        hexes_on_optimal_paths.update(hex for path in optimal_paths_between_ports for hex in path)
    return hexes_on_optimal_paths


def get_non_dead_end_hexes(maze) -> set[int]:
    """
    Given a hex maze, return a set of all hexes on any paths between reward ports.
    Includes all paths between ports (not limited to optimal paths).
    Every other hex is part of a dead end.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[int]: Set of all hexes on paths between reward ports 
            (all hexes that are not part of dead ends)
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    non_dead_end_hexes = set()
    for source_hex, target_hex in [(1, 2), (2, 3), (1, 3)]:
        # Get all possible paths between this pair of reward ports
        all_paths_between_ports = nx.all_simple_paths(graph, source=source_hex, target=target_hex)
        # For each path, add all hexes on the path to our set of non dead end hexes
        non_dead_end_hexes.update(hex for path in all_paths_between_ports for hex in path)
    return non_dead_end_hexes


def get_dead_end_hexes(maze) -> set[int]:
    """
    Given a hex maze, return a set of hexes that are part of dead ends
    (hexes not on any path between reward ports).

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[int]: Set of dead end hexes
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Dead end hexes = (all open hexes) - (non dead end hexes)
    dead_end_hexes = set(graph.nodes) - get_non_dead_end_hexes(graph)
    return dead_end_hexes


def get_non_optimal_non_dead_end_hexes(maze) -> set[int]:
    """
    Given a hex maze, return a set of hexes that are on longer-than-optimal
    (but not dead-end) paths between reward ports. This set may be empty
    if the maze structure is only made up of optimal paths and dead ends.
    (The set of all open hexes in a maze is defined as
    hexes on optimal paths + hexes on non-optimal paths + dead end hexes)

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[int]: Set of hexes on non-optimal paths between reward ports
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Non optimal hexes = (all hexes not in dead ends) - (hexes on optimal paths)
    non_optimal_hexes = get_non_dead_end_hexes(graph) - get_hexes_on_optimal_paths(graph)
    return non_optimal_hexes


def classify_maze_hexes(maze) -> dict[str, object]:
    """
    Given a hex maze, classify hexes as optimal (on optimal paths), 
    non-optimal (on longer-than-optimal but not dead-end paths), 
    and dead-end (on dead-end paths), and return a dictionary with hexes in
    each group and the percentage of hexes in each group.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        dict[str, object]: Dictionary with optimal_hexes, non_optimal_hexes,
            dead_end_hexes, optimal_pct, non_optimal_pct, and dead_end_pct
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Get the set of hexes in each group
    total_hexes = len(graph.nodes)
    optimal_hexes = get_hexes_on_optimal_paths(graph)
    non_optimal_hexes = get_non_optimal_non_dead_end_hexes(graph)
    dead_end_hexes = get_dead_end_hexes(graph)

    return {
        "optimal_hexes": optimal_hexes,
        "optimal_pct": round(len(optimal_hexes) / total_hexes * 100, 2),
        "non_optimal_hexes": non_optimal_hexes,
        "non_optimal_pct": round(len(non_optimal_hexes) / total_hexes * 100, 2),
        "dead_end_hexes": dead_end_hexes,
        "dead_end_pct": round(len(dead_end_hexes) / total_hexes * 100, 2)
    }


def get_dead_ends(maze) -> list[dict[int, int]]:
    """
    Given a hex maze, find all dead end paths. For each dead end,
    return a dictionary where the keys are hexes in that dead end path
    and the values are how far into the dead end each hex is. Note that
    for dead ends that include branches or loops, multiple hexes in the
    dead end will have the same distance.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents
            a distict dead end. Each dictionary entry maps a dead end hex to
            that hex's distance in the dead end.
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Find all hexes that are not part of dead ends
    non_dead_end_hexes = get_non_dead_end_hexes(graph)

    # Get a subgraph of the maze including only hexes that are a part of dead ends
    dead_end_subgraph = graph.subgraph(node for node in graph.nodes if node not in non_dead_end_hexes)

    # Each connected component is a group of dead end hexes
    dead_end_groups = [list(maze_component) for maze_component in nx.connected_components(dead_end_subgraph)]

    dead_end_hex_distances = []
    # For each dead end group, get how far into the dead end each hex is
    for group in dead_end_groups:
        # Get a dict of hex: how far into the dead end it is
        distances = distance_to_nearest_hex_in_group(graph, group, non_dead_end_hexes)
        dead_end_hex_distances.append(distances)
    return dead_end_hex_distances


def get_dead_end_lengths(maze) -> dict[int, int]:
    """
    For each dead end in the maze, find the maximum possible distance into the dead end
    (in hexes), and count how many dead ends have each maximum length

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        dict[int, int]: Dictionary mapping dead end length to the number of dead ends with that length
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Get dead end info (list of dicts for each dead end, each mapping hex to distance in dead end)
    dead_ends = get_dead_ends(graph)

    # Get max distance for each dead end and count number of dead ends with this length
    max_distances = [max(d.values()) for d in dead_ends]
    return dict(sorted(Counter(max_distances).items()))


def get_num_dead_ends(maze, min_length=1) -> int:
    """
    Count the number of dead ends in a maze with minimum length min_length.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        min_length (int): Minimum dead end length to count (default 1)

    Returns:
        int: Number of dead ends in the maze with length >= minimum length
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Get dead end info and count the number of dead ends with length >= minimum
    dead_ends = get_dead_ends(graph)
    return sum(max(d.values()) >= min_length for d in dead_ends)


def is_valid_path(maze, hex_path: list) -> bool:
    """
    Checks if the given hex_path is a valid path through the maze,
    meaning all consecutive hexes exist in the maze and are connected.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        hex_path (list): List of hexes defining a potential path through the maze

    Returns:
        bool: True if the hex_path is valid in the maze, False otherwise.
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # If the path has only one hex, check if it exists in the maze
    if len(hex_path) == 1:
        return hex_path[0] in graph

    # Iterate over consecutive hexes in the path
    for i in range(len(hex_path) - 1):
        # If any consecutive hexes are not connected, the path is invalid
        if not graph.has_edge(hex_path[i], hex_path[i + 1]):
            return False

    return True  # All consecutive hexes exist and are connected


def divide_into_thirds(maze) -> list[set]:
    """
    Given a maze with a single critical choice point, divide the
    open hexes in the maze into 3 sets: hexes between the choice point
    and port 1, hexes between the choice point and port 2, and hexes
    between the choice point and port 3.

    NOT CURRENTLY IMPLEMENTED FOR MAZES WITH MULTIPLE CHOICE POINTS,
    AS DIVIDING THE MAZE INTO 3 GROUPS IS NOT WELL DEFINED IN THIS CASE.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        list[set]: [{hexes between the choice point and port 1},
            {between choice and port 2}, {between choice and port 3}]
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Get choice points for this maze and ensure there is only one
    choice_points = get_critical_choice_points(maze)
    if len(choice_points) != 1:
        raise NotImplementedError(
            f"The given maze has {len(choice_points)} choice points: {choice_points}.\n"
            "This function is only implemented for mazes with a single choice point."
        )

    # Remove the choice point from the maze graph to split it into 3 components
    graph.remove_node(next(iter(choice_points)))

    # Get the 3 components of the split graph (each containing a reward port)
    components = list(nx.connected_components(graph))
    if len(components) != 3:
        print(f"The choice point {choice_points} does not split the maze into 3 distinct components!")
        return None

    # Find each component containing hex 1, hex 2, and hex 3 (in that order)
    thirds = []
    for hex in [1, 2, 3]:
        for maze_component in components:
            if hex in maze_component:
                thirds.append(maze_component)
                break

    return thirds


def get_choice_direction(start_port, end_port) -> str:
    """
    Get the direction of the rat's port choice ('left' or 'right')
    given the rat's start and end port.

    Parameters:
        start_port (int or str): The port the rat started form (1, 2, 3, or A, B, C)
        end_port (int or str): The port the rat ended at (1, 2, 3, or A, B, C)

    Returns:
        str: 'left' or 'right' based on the direction of the rat's choice
    """

    # Create a mapping so we can handle 1, 2, 3 or A, B, C to specify reward ports
    port_hex_map = {"A": 1, "B": 2, "C": 3, 1: 1, 2: 2, 3: 3}
    start = port_hex_map[start_port]
    end = port_hex_map[end_port]

    # Calculate diff mod 3 to handle circular wrapping (1 -> 2 -> 3 in ccw direction)
    diff = (end - start) % 3

    if diff == 1:
        return "right"
    elif diff == 2:
        return "left"
    else:
        # Return None if start_port == end_port
        return None


def has_illegal_straight_path(maze, training_maze=False):
    """
    Given a hex maze, checks if there are any illegal straight paths.
    This criteria differs for regular mazes (max straight path = 6 hexes)
    vs training mazes (max straight path = 8 hexes).

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        training_maze (bool): True if this maze will be used for training,
            meaning the straight path criteria is relaxed slightly. Defaults to False

    Returns:
        The (first) offending path, or False if none
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Get optimal paths between reward ports
    optimal_paths = get_optimal_paths_between_ports(graph)

    # Check if we have any illegal straight paths
    if training_maze:
        # If this is a training maze, use the training maze criteria
        subpaths = set()
        for path in optimal_paths:
            subpaths.update(get_subpaths(path, MAX_STRAIGHT_PATH_TRAINING + 1))
        for path in subpaths:
            if path in ILLEGAL_STRAIGHT_PATHS_TRAINING:
                return path  # (equivalent to returning True)
    else:
        # Otherwise, use the regular criteria
        # We do 2 separate checks here because we may have different
        # path length critera for paths to reward ports vs inside the maze

        # First check all subpaths against illegal paths to a reward port
        subpaths1 = set()
        for path in optimal_paths:
            subpaths1.update(get_subpaths(path, MAX_STRAIGHT_PATH_TO_PORT + 1))
        for path in subpaths1:
            if path in ILLEGAL_STRAIGHT_PATHS_TO_PORT:
                return path  # (equivalent to returning True)

        # Now check all subpaths against illegal paths inside the maze
        subpaths2 = set()
        for path in optimal_paths:
            subpaths2.update(get_subpaths(path, MAX_STRAIGHT_PATH_INSIDE_MAZE + 1))
        for path in subpaths2:
            if path in ILLEGAL_STRAIGHT_PATHS_INSIDE_MAZE:
                return path  # (equivalent to returning True)

    # If we did all of those checks and found no straight paths, we're good to go!
    return False


def is_valid_maze(maze, complain=False) -> bool:
    """
    Given a possible hex maze configuration, check if it is valid using the following criteria:
    - there are no unreachable hexes (this also ensures all reward ports are reachable)
    - path lengths between reward ports are between 15-25 hexes
    - all critical choice points are >=6 hexes away from a reward port
    - there are a maximum of 3 critical choice points
    - no straight paths >MAX_STRAIGHT_PATH_TO_PORT hexes to reward port (including port hex)
    - no straight paths >STRAIGHT_PATHS_INSIDE_MAZE in middle of maze

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        complain (bool): Optional. If our maze configuration is invalid,
            print out the reason why. Defaults to False

    Returns:
        bool: True if the hex maze is valid, False otherwise
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Make sure all (non-barrier) hexes are reachable
    if not nx.is_connected(graph):
        if complain:
            print("BAD MAZE: At least one (non-barrier) hex is unreachable")
        return False

    # Make sure path lengths are between 15-25
    reward_path_lengths = get_reward_path_lengths(graph)
    if min(reward_path_lengths) <= 13:
        if complain:
            print("BAD MAZE: Path between reward ports is too short (<=13)")
        return False
    if max(reward_path_lengths) - min(reward_path_lengths) < 4:
        if complain:
            print("BAD MAZE: Distance difference in reward port paths is too small (<4)")
        return False
    if max(reward_path_lengths) > 25:
        if complain:
            print("BAD MAZE: Path between reward ports is too long (>25)")
        return False

    # Make sure all critical choice points are >=6 hexes away from a reward port
    choice_points = get_critical_choice_points(graph)
    if any(hex in ILLEGAL_CHOICE_POINTS_6 for hex in choice_points):
        if complain:
            print("BAD MAZE: Choice point <6 hexes away from reward port")
        return False

    # Make sure there are not more than 3 critical choice points
    if len(choice_points) > 3:
        if complain:
            print("BAD MAZE: More than 3 critical choice points")
        return False

    # Make sure there are no straight paths
    illegal_path = has_illegal_straight_path(graph)
    if illegal_path:
        if complain:
            print("BAD MAZE: Straight path ", illegal_path)
        return False

    return True


def is_valid_training_maze(maze, complain=False) -> bool:
    """
    Given a possible hex maze configuration, check if it is valid for training using the following criteria:
    - there are no unreachable hexes (this also ensures all reward ports are reachable)
    - all paths between reward ports are the same length
    - path lengths are between 15-23 hexes
    - no straight paths >8 hexes long

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        complain (bool): Optional. If our maze configuration is invalid,
            print out the reason why. Defaults to False

    Returns:
        bool: True if the hex maze is valid, False otherwise
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Make sure all (non-barrier) hexes are reachable
    if not nx.is_connected(graph):
        if complain:
            print("BAD MAZE: At least one (non-barrier) hex is unreachable")
        return False

    # Make sure path lengths are equal and between 15-21 hexes
    reward_path_lengths = get_reward_path_lengths(graph)
    if min(reward_path_lengths) <= 13:
        if complain:
            print("BAD MAZE: Path between reward ports is too short (<=13)")
        return False
    if max(reward_path_lengths) != min(reward_path_lengths):
        if complain:
            print("BAD MAZE: All paths must be the same length")
        return False
    if max(reward_path_lengths) > 23:
        if complain:
            print("BAD MAZE: Path between reward ports is too long (>23)")
        return False

    # Make sure there are no straight paths
    illegal_path = has_illegal_straight_path(graph, training_maze=True)
    if illegal_path:
        if complain:
            print("BAD MAZE: Straight path ", illegal_path)
        return False

    return True


def generate_good_maze(num_barriers=9, training_maze=False) -> set:
    """
    Generates a "good" hex maze as defined by the function is_valid_maze.
    Uses a naive generation approach (randomly generates sets of barriers
    until we get a valid maze configuration).

    Parameters:
        num_barriers (int): How many barriers to place in the maze. Default 9
        training_maze (bool): If this maze is to be used for training,
            meaning it is valid based on a different set of criteria. Uses
            is_valid_training_maze instead of is_valid_maze. Defaults to False

    Returns:
        set: the set of barriers defining the hex maze
    """
    # Create the empty hex maze
    start_maze = create_empty_hex_maze()
    barriers = set()

    # Generate a set of random barriers until we get a good maze
    is_good_maze = False
    while not is_good_maze:
        # Start with an empty hex maze (no barriers)
        test_maze = start_maze.copy()

        # Randomly select some barriers (generally 9 for normal maze, 5/6 for a training maze)
        barriers = set(np.random.choice(POSSIBLE_BARRIERS, size=num_barriers, replace=False))

        # Add the barriers to the empty maze
        for barrier in barriers:
            test_maze.remove_node(barrier)

        # Check if this is a good maze
        is_good_maze = is_valid_maze(test_maze) if not training_maze else is_valid_training_maze(test_maze)

    return barriers



############## Functions for maze rotations and relfections across its axis of symmetry ##############


def rotate_hex(original_hex: int, direction="counterclockwise") -> int:
    """
    Given a hex in the hex maze, returns the corresponding hex if the maze is rotated once
    counterclockwise (e.g. hex 1 becomes hex 2, 4 becomes 49, etc.). Option to specify
    direction='clockwise' to rotate clockwise instead (e.g 1 becomes 3, 4 becomes 48, etc.)

    Parameters:
        original_hex (int): The hex in the hex maze to rotate (1-49)
        direction (str): Which direction to rotate the hex ('clockwise' or 'counterclockwise')
            Defaults to 'counterclockwise'

    Returns:
        int: The corresponding hex if the maze was rotated once in the specified direction
    """
    # Lists of corresponding hexes when the maze is rotated 120 degrees
    hex_rotation_lists = [[1,2,3], [4,49,48], [6,47,33], [5,38,43], [8,42,28], 
                         [7,32,39], [11,46,23], [10,37,34], [9,27,44], [14,41,19],
                         [13,31,29], [12,22,40], [18,45,15], [17,36,24], [16,26,35],
                         [21,30,20], [25]]
    
    for lst in hex_rotation_lists:
        if original_hex in lst:
            index = lst.index(original_hex)
            if direction == "clockwise":
                return lst[(index - 1) % len(lst)]
            else:  # I choose to assume any direction not specified 'clockwise' is 'counterclockwise'
                return lst[(index + 1) % len(lst)]
    # Return None if the hex to rotate doesn't exist in our rotation lists (all hexes should exist)
    return None


def reflect_hex(original_hex: int, axis=1) -> int:
    """
    Given a hex in the hex maze, returns the corresponding hex if the maze is reflected
    across the axis of hex 1 (e.g. hex 6 becomes hex 5 and vice versa, 8 becomes 7, etc.).
    Option to specify axis=2 or axis=3 to reflect across the axis of hex 2 or 3 instead.

    Parameters:
        original_hex (int): The hex in the maze to reflect (1-49)
        axis (int): Which reward port axis to reflect the maze across. Must be
            1, 2, or 3. Defaults to 1

    Returns:
        int: The corresponding hex if the maze was reflected across the specified axis
    """
    # Lists of corresponding hexes reflected across axis 1, 2, or 3
    reflections_ax1 = [[6,5], [8,7], [11,9], [14,12], [18,15], [17,16], [22,19], 
                      [21,20], [27,23], [26,24], [32,28], [31,29], [38,33], [37,34],
                      [36,35], [49,48], [42,39], [41,40],[2,3], [47,43], [46,44]]
    reflections_ax2 = [[47,38], [42,32], [46,27], [41,22], [45,18], [36,26], [40,14], 
                      [30,21], [44,11], [35,17], [39,8], [29,13], [43,6], [34,10], 
                      [24,16], [48,4], [28,7], [19,12], [3,1], [33,5], [23,9]]
    reflections_ax3 = [[43,33], [39,28], [44,23], [40,19], [45,15], [35,24], [41,12],
                      [30,20], [46,9], [36,16], [42,7], [31,13], [47,5], [37,10],
                      [26,17], [49,4], [32,8], [22,14], [2,1], [38,6], [27,11]]
    # Choose the reflection list for the axis we care about
    hex_reflections = {1: reflections_ax1, 2: reflections_ax2, 3: reflections_ax3}.get(axis, None)

    for lst in hex_reflections:
        if original_hex in lst:
            # Return the other hex in the reflection pair
            return lst[1] if lst[0] == original_hex else lst[0]
    # If the hex isn't in any list, it doesn't change when the maze is reflected along this axis
    return original_hex


def get_rotated_barriers(original_barriers, direction="counterclockwise") -> set:
    """
    Given a hex maze, returns the corresponding barrier set if the maze is rotated
    once counterclockwise (e.g. hex 1 becomes hex 2, 4 becomes 49, etc.).
    Option to specify direction='clockwise' to rotate clockwise
    instead (e.g 1 becomes 3, 4 becomes 48, etc.)

    Parameters:
        original_barriers (list, set, frozenset, np.ndarray, str, nx.Graph):
            The original hex maze, represented in any valid format
        direction (str): Which direction to rotate the maze ('clockwise' or 'counterclockwise')
            Defaults to 'counterclockwise'

    Returns:
        set: The barrier set if the maze was rotated once in the specified direction
    """
    # Convert all valid maze representations to a set of ints representing barrier hexes
    original_barriers = maze_to_barrier_set(original_barriers)

    return {rotate_hex(b, direction) for b in original_barriers}


def get_reflected_barriers(original_barriers, axis=1) -> set:
    """
    Given a hex maze, returns the corresponding barrier set if the maze is reflected
    along the axis of hex 1 (e.g. hex 6 becomes hex 5 and vice versa, 8 becomes 7 and vice versa, etc.).
    Option to specify axis=2 or axis=3 to reflect across the axis of hex 2 or 3 instead.

    Parameters:
        original_barriers (list, set, frozenset, np.ndarray, str, nx.Graph):
            The original hex maze, represented in any valid format
        axis (int): Which reward port axis to reflect the maze across.
            Must be 1, 2, or 3. Defaults to 1

    Returns:
        set: The barrier set if the maze was reflected across the specified axis
    """
    # Convert all valid maze representations to a set of ints representing barrier hexes
    original_barriers = maze_to_barrier_set(original_barriers)

    return {reflect_hex(b, axis) for b in original_barriers}


def get_isomorphic_mazes(maze) -> set[frozenset]:
    """
    Given a hex maze, return the other 5 mazes that have the same graph structure
    (corresponding to the maze rotated clockwise/counterclockwise and
    reflected across its 3 axes of symmetry)

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[frozenset]: a set of the 5 barrier sets defining mazes isomorphic to this maze
    """
    # Rotate and reflect the maze to get other barrier configs that
    # represent the same underlying graph structure
    reflected_ax1 = frozenset(get_reflected_barriers(maze, axis=1))
    reflected_ax2 = frozenset(get_reflected_barriers(maze, axis=2))
    reflected_ax3 = frozenset(get_reflected_barriers(maze, axis=3))
    rotated_ccw = frozenset(get_rotated_barriers(maze, direction="counterclockwise"))
    rotated_cw = frozenset(get_rotated_barriers(maze, direction="clockwise"))

    return {reflected_ax1, reflected_ax2, reflected_ax3, rotated_ccw, rotated_cw}


def get_maze_attributes(maze) -> dict:
    """
    Given a hex maze, create a dictionary of attributes for that maze.
    Includes the length of the optimal paths between reward ports, the optimal paths
    between these ports, the path length difference between optimal paths,
    critical choice points, the hexes (and percentage of hexes) that are on optimal
    paths, non-optimal paths, or in dead ends, the number of dead ends of each length,
    the number of cycles and the hexes defining these cycles,
    and a set of other maze configurations isomorphic to this maze.

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        dict: A dictionary of attributes of this maze
    """
    # Convert all valid maze representations to a nx.Graph object
    graph = maze_to_graph(maze)

    # Convert all valid maze representations to a set of ints representing barrier hexes
    barriers = maze_to_barrier_set(maze)

    # Get length of optimal paths between reward ports
    len12 = nx.shortest_path_length(graph, source=1, target=2) + 1
    len13 = nx.shortest_path_length(graph, source=1, target=3) + 1
    len23 = nx.shortest_path_length(graph, source=2, target=3) + 1
    reward_path_lengths = [len12, len13, len23]
    path_length_difference = max(reward_path_lengths) - min(reward_path_lengths)

    # Get the optimal paths between reward ports
    optimal_paths_12 = list(nx.all_shortest_paths(graph, source=1, target=2))
    optimal_paths_13 = list(nx.all_shortest_paths(graph, source=1, target=3))
    optimal_paths_23 = list(nx.all_shortest_paths(graph, source=2, target=3))
    optimal_paths_all = []
    optimal_paths_all.extend(optimal_paths_12)
    optimal_paths_all.extend(optimal_paths_13)
    optimal_paths_all.extend(optimal_paths_23)

    # Get critical choice points
    choice_points = set(get_critical_choice_points(graph))
    num_choice_points = len(choice_points)

    # Get information about cycles
    cycle_basis = nx.cycle_basis(graph)
    num_cycles = len(cycle_basis)

    # Get a list of isomorphic mazes
    isomorphic_mazes = get_isomorphic_mazes(barriers)

    # Classify hexes as on optimal path, on non-optimal path, or dead end
    hex_type_dict = classify_maze_hexes(graph)

    # Get dead end lengths
    # Note: this makes including num_dead_ends_min_length_X superfluous, but we keep
    # both because it's neater to query the database using num_dead_ends_min_length_X
    dead_end_lengths = get_dead_end_lengths(graph)

    # Create a dictionary of attributes
    attributes = {
        "barriers": barriers,
        "len12": len12,
        "len13": len13,
        "len23": len23,
        "reward_path_lengths": reward_path_lengths,
        "path_length_difference": path_length_difference,
        "optimal_pct": hex_type_dict["optimal_pct"],
        "non_optimal_pct": hex_type_dict["non_optimal_pct"],
        "dead_end_pct": hex_type_dict["dead_end_pct"],
        "optimal_hexes": hex_type_dict["optimal_hexes"],
        "non_optimal_hexes": hex_type_dict["non_optimal_hexes"],
        "dead_end_hexes": hex_type_dict["dead_end_hexes"],
        "dead_end_lengths": dead_end_lengths,
        "num_dead_ends": get_num_dead_ends(graph),
        "num_dead_ends_min_length_2": get_num_dead_ends(graph, min_length=2),
        "num_dead_ends_min_length_3": get_num_dead_ends(graph, min_length=3),
        "num_dead_ends_min_length_4": get_num_dead_ends(graph, min_length=4),
        "optimal_paths_12": optimal_paths_12,
        "optimal_paths_13": optimal_paths_13,
        "optimal_paths_23": optimal_paths_23,
        "optimal_paths_all": optimal_paths_all,
        "choice_points": choice_points,
        "num_choice_points": num_choice_points,
        "cycles": cycle_basis,
        "num_cycles": num_cycles,
        "isomorphic_mazes": isomorphic_mazes,
    }
    return attributes


def get_barrier_sequence_attributes(barrier_sequence: list[set]) -> dict:
    """
    Given a sequence of maze configurations that differ by the movement of a single barrier,
    get the barrier change between each maze, reward path lengths, and
    choice points for all mazes in the sequence, and return a dictionary of these attributes.

    Parameters:
        barrier_sequence (list[set]): The sequence of maze configurations.

    Returns:
        dict: A dictionary of attributes of this sequence.
    """

    reward_path_lengths = []
    choice_points = []

    # Get attributes for each barrier set in the sequence
    for bars in barrier_sequence:
        reward_path_lengths.append(get_reward_path_lengths(bars))
        choice_points.append(get_critical_choice_points(bars))

    barrier_changes = get_barrier_changes(barrier_sequence)

    # Set up a dictionary of attributes
    barrier_dict = {
        "barrier_sequence": barrier_sequence,
        "sequence_length": len(barrier_sequence),
        "barrier_changes": barrier_changes,
        "reward_path_lengths": reward_path_lengths,
        "choice_points": choice_points,
    }
    return barrier_dict

