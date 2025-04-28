import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import math
from itertools import chain
from scipy.spatial import KDTree
from collections import Counter


# for now this is defined here because we use it to set up constants
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

################################# Define a bunch of functions #################################


############## Helper functions for spyglass compatibility ##############


def set_to_string(barrier_set: set[int]) -> str:
    """
    Converts a set of ints to a sorted, comma-separated string.
    Used for going from a set of barrier locations to a query-able config_id
    for compatibility with HexMazeConfig in spyglass.

    Parameters:
        barrier_set (set[int]): A set of ints representing where barriers are placed in the maze

    Returns:
        str: A sorted, comma-separated string representing where barriers are placed in the maze
    """
    return ",".join(map(str, sorted(barrier_set)))


def string_to_set(string: str) -> set[int]:
    """
    Converts a sorted, comma-separated string (used as a config_id for the
    HexMazeConfig in spyglass) to a set of ints (for compatability with hex maze functions)

    Parameters:
        string (str): A sorted, comma-separated string representing where barriers are placed in the maze

    Returns:
        set[int]: A set of ints representing where barriers are placed in the maze
    """
    string = string.strip("{}[]()")  # strip just in case, to handle more variable inputs
    return set(map(int, string.split(",")))


############## Functions for generating a hex maze configuration ##############


def add_edges_to_node(graph: nx.Graph, node: int, edges: list[int]):
    """
    Add all edges to the specified node in the graph.
    If the node does not yet exist in the graph, add the node.
    Modifies the graph in-place.

    Parameters:
        graph (nx.Graph): The networkx graph object
        node (int): The node to add to the graph (if it does not yet exist)
        edges (list[int]): The edges to the node in the graph
    """
    for edge in edges:
        graph.add_edge(node, edge)


def create_empty_hex_maze() -> nx.Graph:
    """
    Use networkx to create a graph object representing the empty hex maze 
    before any barriers are added.

    Returns: 
        nx.Graph: A new networkx graph object representing all of the hexes \
            and potential transitions between them in the hex maze
    """
    empty_hex_maze = nx.Graph()

    # Define all nodes and edges to create the empty maze
    add_edges_to_node(empty_hex_maze, 1, [4])
    add_edges_to_node(empty_hex_maze, 4, [1, 5, 6])
    add_edges_to_node(empty_hex_maze, 6, [4, 8])
    add_edges_to_node(empty_hex_maze, 5, [4, 7])
    add_edges_to_node(empty_hex_maze, 8, [6, 11, 10])
    add_edges_to_node(empty_hex_maze, 7, [5, 10, 9])
    add_edges_to_node(empty_hex_maze, 11, [8, 14])
    add_edges_to_node(empty_hex_maze, 10, [8, 7, 13])
    add_edges_to_node(empty_hex_maze, 9, [7, 12])
    add_edges_to_node(empty_hex_maze, 14, [11, 18, 17])
    add_edges_to_node(empty_hex_maze, 13, [10, 17, 16])
    add_edges_to_node(empty_hex_maze, 12, [9, 16, 15])
    add_edges_to_node(empty_hex_maze, 18, [14, 22])
    add_edges_to_node(empty_hex_maze, 17, [14, 13, 21])
    add_edges_to_node(empty_hex_maze, 16, [13, 12, 20])
    add_edges_to_node(empty_hex_maze, 15, [12, 19])
    add_edges_to_node(empty_hex_maze, 22, [18, 27, 26])
    add_edges_to_node(empty_hex_maze, 21, [17, 26, 25])
    add_edges_to_node(empty_hex_maze, 20, [16, 25, 24])
    add_edges_to_node(empty_hex_maze, 19, [15, 24, 23])
    add_edges_to_node(empty_hex_maze, 27, [22, 32])
    add_edges_to_node(empty_hex_maze, 26, [22, 21, 31])
    add_edges_to_node(empty_hex_maze, 25, [21, 20, 30])
    add_edges_to_node(empty_hex_maze, 24, [20, 19, 29])
    add_edges_to_node(empty_hex_maze, 23, [19, 28])
    add_edges_to_node(empty_hex_maze, 32, [27, 38, 37])
    add_edges_to_node(empty_hex_maze, 31, [26, 37, 36])
    add_edges_to_node(empty_hex_maze, 30, [25, 36, 35])
    add_edges_to_node(empty_hex_maze, 29, [24, 35, 34])
    add_edges_to_node(empty_hex_maze, 28, [23, 34, 33])
    add_edges_to_node(empty_hex_maze, 38, [32, 49])
    add_edges_to_node(empty_hex_maze, 37, [31, 32, 42])
    add_edges_to_node(empty_hex_maze, 36, [30, 31, 41])
    add_edges_to_node(empty_hex_maze, 35, [29, 30, 40])
    add_edges_to_node(empty_hex_maze, 34, [28, 29, 39])
    add_edges_to_node(empty_hex_maze, 33, [28, 48])
    add_edges_to_node(empty_hex_maze, 49, [2, 38, 47])
    add_edges_to_node(empty_hex_maze, 42, [37, 46, 47])
    add_edges_to_node(empty_hex_maze, 41, [36, 45, 46])
    add_edges_to_node(empty_hex_maze, 40, [35, 44, 45])
    add_edges_to_node(empty_hex_maze, 39, [34, 43, 44])
    add_edges_to_node(empty_hex_maze, 48, [3, 33, 43])
    add_edges_to_node(empty_hex_maze, 2, [49])
    add_edges_to_node(empty_hex_maze, 47, [49, 42])
    add_edges_to_node(empty_hex_maze, 46, [42, 41])
    add_edges_to_node(empty_hex_maze, 45, [41, 40])
    add_edges_to_node(empty_hex_maze, 44, [40, 39])
    add_edges_to_node(empty_hex_maze, 43, [39, 48])
    add_edges_to_node(empty_hex_maze, 3, [48])
    return empty_hex_maze


def create_maze_graph(barrier_set: set[int]) -> nx.Graph:
    """
    Given a set of barriers defining a hex maze configuration,
    return a networkx graph object representing the maze.

    Parameters:
        barrier_set (set[int]): Set of hex locations
            where barriers are placed in this hex maze configuration

    Returns:
        nx.Graph: A networkx graph object representing the maze
    """

    # Create a new empty hex maze object
    maze_graph = create_empty_hex_maze()

    # Remove the barriers
    for barrier in barrier_set:
        maze_graph.remove_node(barrier)
    return maze_graph


def maze_to_graph(maze) -> nx.Graph:
    """
    Converts a hex maze represented in any valid format (list, set, frozenset, numpy
    array, string, or networkx graph) to a networkx graph object representing the maze.

    Processes the following maze formats:
        - list: A list of barrier hexes
        - set/frozenset: A set of barrier hexes
        - numpy array: A 1D numpy array of barrier hexes
        - str: A comma-separated string of barrier hexes
        - nx.Graph: A networkx graph object representing the maze structure

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        nx.Graph: A networkx graph object representing the maze
    """

    if isinstance(maze, str):
        # Convert string to a set of barriers
        maze = string_to_set(maze)
    if isinstance(maze, (set, frozenset, list, np.ndarray)):
        if isinstance(maze, np.ndarray):
            if maze.ndim != 1:
                raise ValueError(f"Expected 1D array of barriers, got shape {maze.shape}")
            maze = list(maze)
        # Convert barrier set to a graph
        return create_maze_graph(maze)
    elif isinstance(maze, nx.Graph):
        # If it's already a graph, use that (but return a copy to avoid modifying the original)
        return maze.copy()
    raise TypeError(f"Expected maze to be a set, frozenset, list, 1D numpy array, or nx.Graph, got {type(maze)}")


def maze_to_barrier_set(maze) -> set[int]:
    """
    Converts a hex maze represented in any valid format (list, set, frozenset, numpy
    array, string, or networkx graph) to a set of barrier locations.

    Processes the following maze formats:
        - list: A list of barrier hexes
        - set/frozenset: A set of barrier hexes
        - numpy array: A 1D numpy array of barrier hexes
        - str: A comma-separated string of barrier hexes
        - nx.Graph: A networkx graph object representing the maze structure

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        set[int]: A set of hexes where barriers are placed in the maze
    """
    # If it's already a set/list/frozenset/array, just process it directly
    if isinstance(maze, (set, frozenset, list, np.ndarray)):
        if isinstance(maze, np.ndarray):
            if maze.ndim != 1:
                raise ValueError(f"Expected 1D array of barriers, got shape {maze.shape}")
            maze = list(maze)  # Convert to list for consistency
        barrier_set = set(maze)

    # If it's a string, convert to a set
    elif isinstance(maze, str):
        barrier_set = string_to_set(maze)

    # If it's a graph, the barrier locations are the nodes not present in the maze graph
    elif isinstance(maze, nx.Graph):
        all_possible_hexes = set(range(1, 50))
        open_hexes = set(maze.nodes)
        barrier_set = all_possible_hexes - open_hexes

    else:
        raise TypeError(f"Unsupported maze format: {type(maze)}")

    # Make sure it's a set of ints (not int64)
    barrier_set = {int(x) for x in barrier_set}
    return barrier_set


def maze_to_string(maze) -> str:
    """
    Converts a hex maze represented in any valid format (list, set, frozenset, numpy
    array, string, or networkx graph) to a sorted, comma-separated string.
    Used for converting to a query-able config_id for compatibility with HexMazeConfig in spyglass.

    Processes the following maze formats:
        - list: A list of barrier hexes
        - set/frozenset: A set of barrier hexes
        - numpy array: A 1D numpy array of barrier hexes
        - str: A comma-separated string of barrier hexes
        - nx.Graph: A networkx graph object representing the maze structure

    Parameters:
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format

    Returns:
        str: A sorted, comma-separated string representing where barriers are placed in the maze
    """

    # Convert all valid maze representations to a set of ints representing barrier hexes
    barrier_set = maze_to_barrier_set(maze)

    # Convert barrier set to string
    return set_to_string(barrier_set)


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


############## Functions for generating a next good barrier set given an initial barrier set ##############


def single_barrier_moved(maze_1, maze_2) -> bool:
    """
    Check if two hex mazes differ by the movement of a single barrier.

    This means the number of barriers is the same in both mazes, and to go
    from one maze to the other, exactly one barrier changes location.

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        bool: True if the mazes differ by the movement of a single barrier, False otherwise
    """
    # Convert all valid maze representations to a set of ints representing barrier hexes
    maze_1_barrier_set = maze_to_barrier_set(maze_1)
    maze_2_barrier_set = maze_to_barrier_set(maze_2)

    # Check that both mazes have the same number of barriers
    if len(maze_1_barrier_set) != len(maze_2_barrier_set):
        return False

    # The symmetric difference (XOR) between the sets must have exactly two elements
    # because each set should have exactly one barrier not present in the other set
    return len(maze_1_barrier_set.symmetric_difference(maze_2_barrier_set)) == 2


def have_common_path(paths_1: list[list], paths_2: list[list]) -> bool:
    """
    Given 2 lists of hex paths, check if there is a common path between the 2 lists.
    Used for determining if there are shared optimal paths between mazes.

    Parameters:
        paths_1 (list[list]): List of optimal hex paths between 2 reward ports
        paths_2 (list[list]): List of optimal hex paths between 2 reward ports

    Returns:
        bool: True if there is a common path between the 2 lists of paths, False otherwise.
    """

    # Convert the path lists to tuples to make them hashable and store them in sets
    pathset_1 = set(tuple(path) for path in paths_1)
    pathset_2 = set(tuple(path) for path in paths_2)

    # Return True if there is 1 or more common path between the path sets, False otherwise
    return len(pathset_1.intersection(pathset_2)) > 0


def have_common_optimal_paths(maze_1, maze_2) -> bool:
    """
    Given 2 hex mazes, check if the 2 mazes have at least one common optimal path
    between every pair of reward ports (e.g. the mazes share an optimal path between
    ports 1 and 2, AND ports 1 and 3, AND ports 2 and 3), meaning the rat could be
    running the same paths even though the mazes are "different".

    (The result of this function is equivalent to checking if num_hexes_different_on_optimal_paths == 0)

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        bool: True if the mazes have a common optimal path between all pairs of reward ports, False otherwise
    """
    # Do these barrier sets have a common optimal path from port 1 to port 2?
    have_common_path_12 = have_common_path(
        get_optimal_paths(maze_1, start_hex=1, target_hex=2), get_optimal_paths(maze_2, start_hex=1, target_hex=2)
    )
    # Do these barrier sets have a common optimal path from port 1 to port 3?
    have_common_path_13 = have_common_path(
        get_optimal_paths(maze_1, start_hex=1, target_hex=3), get_optimal_paths(maze_2, start_hex=1, target_hex=3)
    )
    # Do these barrier sets have a common optimal path from port 2 to port 3?
    have_common_path_23 = have_common_path(
        get_optimal_paths(maze_1, start_hex=2, target_hex=3), get_optimal_paths(maze_2, start_hex=2, target_hex=3)
    )

    # Return True if the barrier sets have a common optimal path between all pairs of reward ports
    return have_common_path_12 and have_common_path_13 and have_common_path_23


def min_hex_diff_between_paths(paths_1: list[list], paths_2: list[list]) -> int:
    """
    Given 2 lists of hex paths, return the minimum number of hexes that differ
    between the most similar paths in the 2 lists.
    Used for determining how different optimal paths are between mazes.

    Parameters:
        paths_1 (list[list]): List of optimal hex paths (usually between 2 reward ports)
        paths_2 (list[list]): List of optimal hex paths (usually between 2 reward ports)

    Returns:
        num_different_hexes (int): the min number of hexes different between a
            hex path in paths_1 and a hex path in paths_2 (hexes on path1 not on path2 +
            hexes on path2 not on path1). If there is 1 or more shared
            path between the path lists, the hex difference is 0.
    """

    # If there is 1 or more shared path between the path sets, the hex difference is 0
    if have_common_path(paths_1, paths_2):
        return 0

    # Max possible number of different hexes between paths
    num_different_hexes = 25

    for path_a in paths_1:
        for path_b in paths_2:
            # Get how many hexes differ between these paths
            diff = len(set(path_a).symmetric_difference(set(path_b)))
            # Record the minimum possible difference between optimal paths
            if diff < num_different_hexes:
                num_different_hexes = diff

    return num_different_hexes


def hexes_different_between_paths(paths_1: list[list], paths_2: list[list]) -> tuple[set, set]:
    """
    Given 2 lists of hex paths, identify hexes that differ between the most similar
    paths in each list. Used for determining how different optimal paths are between mazes.

    The function finds the pair of paths (one from each list) that are most similar,
    then returns:
      - The set of hexes in the first path but not the second
      - The set of hexes in the second path but not the first

    If there is one or more path that appears in both lists, both sets will be empty.

    Parameters:
        paths_1 (list[list]): List of optimal hex paths (usually between 2 reward ports)
        paths_2 (list[list]): List of optimal hex paths (usually between 2 reward ports)

    Returns:
        tuple[set, set]:
            - Set of hexes on a path from paths_1 but not on the most similar path from paths_2
            - Set of hexes on a path from paths_2 but not on the most similar path from paths_1
    """
    hexes_on_path_1_not_path_2 = set()
    hexes_on_path_2_not_path_1 = set()

    # If there is 1 or more shared path between the path sets, the hex difference is 0
    if have_common_path(paths_1, paths_2):
        return hexes_on_path_1_not_path_2, hexes_on_path_2_not_path_1

    # Max possible number of different hexes between paths
    num_different_hexes = 25

    # Find the most similar path between the path lists, and get the different hexes on it
    for path_a in paths_1:
        for path_b in paths_2:
            # Get how many hexes differ between these paths
            diff = len(set(path_a).symmetric_difference(set(path_b)))
            # Record the minimum possible difference between optimal paths
            if diff < num_different_hexes:
                num_different_hexes = diff
                hexes_on_path_1_not_path_2 = set([hex for hex in path_a if hex not in path_b])
                hexes_on_path_2_not_path_1 = set([hex for hex in path_b if hex not in path_a])

    return hexes_on_path_1_not_path_2, hexes_on_path_2_not_path_1


def hexes_different_on_optimal_paths(maze_1, maze_2) -> tuple[set, set]:
    """
    Given 2 hex mazes, find the set of hexes different on optimal paths between
    every pair of reward ports. This helps us quantify how different two maze configurations are.

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        tuple[set, set]:
            - Hexes uniquely on optimal paths in maze_1 (not in maze_2)
            - Hexes uniquely on optimal paths in maze_2 (not in maze_1)
    """

    # Get which hexes are different on the most similar optimal paths from port 1 to port 2
    maze1_hexes_path12, maze2_hexes_path12 = hexes_different_between_paths(
        get_optimal_paths(maze_1, start_hex=1, target_hex=2), get_optimal_paths(maze_2, start_hex=1, target_hex=2)
    )
    # Get which hexes are different on the most similar optimal paths from port 1 to port 3
    maze1_hexes_path13, maze2_hexes_path13 = hexes_different_between_paths(
        get_optimal_paths(maze_1, start_hex=1, target_hex=3), get_optimal_paths(maze_2, start_hex=1, target_hex=3)
    )
    # Get which hexes are different on the most similar optimal paths from port 2 to port 3
    maze1_hexes_path23, maze2_hexes_path23 = hexes_different_between_paths(
        get_optimal_paths(maze_1, start_hex=2, target_hex=3), get_optimal_paths(maze_2, start_hex=2, target_hex=3)
    )

    # Get the combined set of hexes different between the most similar optimal
    # paths between all 3 reward ports
    hexes_on_optimal_paths_maze_1_not_2 = maze1_hexes_path12 | maze1_hexes_path13 | maze1_hexes_path23
    hexes_on_optimal_paths_maze_2_not_1 = maze2_hexes_path12 | maze2_hexes_path13 | maze2_hexes_path23
    # Return hexes exlusively on optimal paths in maze 1, and hexes exclusively on optimal paths in maze 2
    return hexes_on_optimal_paths_maze_1_not_2, hexes_on_optimal_paths_maze_2_not_1


def num_hexes_different_on_optimal_paths(maze_1, maze_2) -> int:
    """
    Given 2 hex mazes, find the number of hexes different on the optimal
    paths between every pair of reward ports. This difference is equal to
    (number of hexes on optimal paths in maze_1 but not maze_2 +
    number of hexes on optimal paths in maze_2 but not maze_1)

    This helps us quantify how different two maze configurations are.

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        num_different_hexes (int): The number of hexes different on optimal paths
            between reward ports for maze_1 and maze_2
    """

    # Get the hexes different on optimal paths between these 2 mazes
    hexes_maze1_not_maze2, hexes_maze2_not_maze1 = hexes_different_on_optimal_paths(maze_1, maze_2)

    # Return the combined number of hexes different on the optimal paths
    return len(hexes_maze1_not_maze2 | hexes_maze2_not_maze1)


def num_hexes_different_on_optimal_paths_isomorphic(maze_1, maze_2, mode="all"):
    """
    Given 2 hex mazes, find the number of hexes different on the optimal
    paths between every pair of reward ports, for all isomorphic versions of the mazes.
    This difference is equal to (number of hexes on optimal paths in maze_1 but not maze_2 +
    number of hexes on optimal paths in maze_2 but not maze_1).
    Like 'num_hexes_different_on_optimal_paths', but checks against
    all isomorphic configurations (checks maze similarity against rotated
    and flipped versions of these hex mazes)

    This helps us quantify how different two maze configurations are.

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format
        mode (str): Type of isomorphic mazes to check:
            'all' = all isomorphic mazes, both rotations and flips (default)
            'rotation' = only rotations
            'reflection' or 'flip' = only reflections

    Returns:
        min_num_different_hexes (int): The minimum number of hexes different on
            optimal paths between reward ports for all isomorphic versions of maze_1 and maze_2
        most_similar_maze (set): The (rotated, flipped) version of maze_1 that is most similar to maze_2
    """

    # Start by comparing the normal versions of maze_1 and maze_2
    min_num_different_hexes = num_hexes_different_on_optimal_paths(maze_1, maze_2)
    # Also track which version of maze_1 is most similar to maze_2
    most_similar_maze = maze_1

    mode = mode.lower()
    isomorphic_mazes = []

    # If we only care about rotations, only add those to the comparison list
    if mode == "rotation":
        isomorphic_mazes.append(get_rotated_barriers(maze_1, direction="clockwise"))
        isomorphic_mazes.append(get_rotated_barriers(maze_1, direction="counterclockwise"))

    # Or if we only care about reflections, only add those
    elif mode in {"reflection", "flip"}:
        isomorphic_mazes.append(get_reflected_barriers(maze_1, axis=1))
        isomorphic_mazes.append(get_reflected_barriers(maze_1, axis=2))
        isomorphic_mazes.append(get_reflected_barriers(maze_1, axis=3))

    # Otherwise, consider all isomorphic mazes
    else:
        isomorphic_mazes = list(get_isomorphic_mazes(maze_1))

    # Compare each isomorphic maze with maze_2 and update the minimum number of different hexes
    for iso_maze_1 in isomorphic_mazes:
        num_different_hexes = num_hexes_different_on_optimal_paths(iso_maze_1, maze_2)
        if num_different_hexes < min_num_different_hexes:
            min_num_different_hexes = num_different_hexes
            most_similar_maze = iso_maze_1

    return min_num_different_hexes, most_similar_maze


def at_least_one_path_shorter_and_longer(maze_1, maze_2) -> bool:
    """
    Given 2 hex mazes, check if at least one optimal path between reward ports
    is shorter AND at least one is longer in one of the mazes compared to the other
    (e.g. the path length between ports 1 and 2 increases and the path length
    between ports 2 and 3 decreases.

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        bool: True if at least one path is shorter AND at least one is longer, False otherwise
    """
    # Get path lengths between reward ports for each barrier set
    paths_1 = get_reward_path_lengths(maze_1)
    paths_2 = get_reward_path_lengths(maze_2)

    # Check if >=1 path is longer and >=1 path is shorter
    return any(a < b for a, b in zip(paths_1, paths_2)) and any(a > b for a, b in zip(paths_1, paths_2))


def optimal_path_order_changed(maze_1, maze_2) -> bool:
    """
    Given 2 hex mazes, check if the length order of the optimal paths
    between reward ports has changed (e.g. the shortest path between reward ports
    used to be between ports 1 and 2 and is now between ports 2 and 3, etc.)

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        bool: True if the optimal path length order has changed, False otherwise
    """

    # Get path lengths between reward ports for each barrier set
    paths_1 = get_reward_path_lengths(maze_1)
    paths_2 = get_reward_path_lengths(maze_2)

    # Find which are the longest and shortest paths (multiple paths may tie for longest/shortest)
    longest_paths_1 = [i for i, num in enumerate(paths_1) if num == max(paths_1)]
    shortest_paths_1 = [i for i, num in enumerate(paths_1) if num == min(paths_1)]
    longest_paths_2 = [i for i, num in enumerate(paths_2) if num == max(paths_2)]
    shortest_paths_2 = [i for i, num in enumerate(paths_2) if num == min(paths_2)]

    # Check that both the longest and shortest paths are not the same
    return not any(l in longest_paths_2 and s in shortest_paths_2 for l in longest_paths_1 for s in shortest_paths_1)


def no_common_choice_points(maze_1, maze_2) -> bool:
    """
    Given 2 mazes, check that there are no common critical choice points between them.

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        bool: True if there are no common choice points, False otherwise
    """

    # Get the choice points for each barrier set
    choice_points_1 = get_critical_choice_points(maze_1)
    choice_points_2 = get_critical_choice_points(maze_2)

    # Check if there are no choice points in common
    return choice_points_1.isdisjoint(choice_points_2)


def get_barrier_change(maze_1, maze_2) -> tuple[int, int]:
    """
    Given 2 hex mazes that differ by the movement of a single barrier,
    find the barrier that was moved.

     Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format

    Returns:
        old_barrier (int): The hex location of the barrier to be moved in the first set
        new_barrier (int): The hex location the barrier was moved to in the second set

    Raises:
        ValueError: If the mazes do not differ by exactly one moved barrier
    """
    # Convert all valid maze representations to a set of ints representing barrier hexes
    maze_1_barrier_set = maze_to_barrier_set(maze_1)
    maze_2_barrier_set = maze_to_barrier_set(maze_2)

    # Enforce that the 2 mazes differ by the movement of a single barrier?
    if not single_barrier_moved(maze_1, maze_2):
        raise ValueError("maze_1 and maze_2 must differ by the movement of a single barrier!")

    # Find the original barrier location
    old_barrier = maze_1_barrier_set - maze_2_barrier_set

    # Find the new barrier location
    new_barrier = maze_2_barrier_set - maze_1_barrier_set

    # Return as integers instead of sets/frozensets with a single element
    return next(iter(old_barrier)), next(iter(new_barrier))


def get_barrier_changes(barrier_sequence: list[set]) -> list[list]:
    """
    Given a sequence of barrier sets that each differ by the movement of
    a single barrier, find the barriers moved from each barrier set to the next.

    Parameters:
        barrier_sequence (list[set]): List of sequential barrier sets

    Returns:
        list[list]: A list of [old barrier, new barrier] defining each transition between barrier sets
    """
    barrier_changes = []
    for i in range(len(barrier_sequence) - 1):
        old_barrier, new_barrier = get_barrier_change(barrier_sequence[i], barrier_sequence[i + 1])
        barrier_changes.append([old_barrier, new_barrier])
    return barrier_changes


def get_next_barrier_sets(df: pd.DataFrame, original_barriers, criteria_type="ALL") -> list[set]:
    """
    Given the hex maze database (df) and a starting maze, get a list
    of next barrier sets created by the movement of a single barrier.

    We have 2 criteria:
    1. At least one path must be longer and one must be shorter.
    2. The optimal path order must have changed (the pair of reward ports that
    used to be the closest together or furthest apart is now different).

    Parameters:
        df (pd.DataFrame): Database of potential hex maze configurations we want to search from
        original_barriers (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze to start from, represented in any valid format
        criteria_type (str):
            'ANY': (default) Accept new barrier sets that meet EITHER of these criteria
            'ALL': Accept new barrier sets that meet BOTH of these criteria
            'JOSE': Meets both of the above criteria, AND optimal path lengths are 17, 19, 21,
                AND only 1 choice point

    Returns:
        list[set]: A list of potential new barrier sets
    """
    # Convert all valid maze representations to a set of ints representing barrier hexes
    original_barriers = maze_to_barrier_set(original_barriers)

    # Find other valid mazes in the df that differ by the movement of a single barrier
    potential_new_barriers = [b for b in df["barriers"] if single_barrier_moved(b, original_barriers)]

    # Set up a list for the ones that meet our criteria
    new_barriers = []

    # Check each potential new barrier set
    for bar in potential_new_barriers:
        # Check if at least one path gets longer and at least one path gets shorter
        criteria1 = at_least_one_path_shorter_and_longer(original_barriers, bar)
        # Check if the optimal path order has changed
        criteria2 = optimal_path_order_changed(original_barriers, bar)
        # Make sure the optimal path lengths are 17, 19, 21 (in any order)
        criteria3 = set(df_lookup(df, bar, "reward_path_lengths")) == {17, 19, 21}
        # Only 1 critical choice point
        criteria4 = df_lookup(df, bar, "num_choice_points") == 1

        # Accept the potential new barrier set if it meets our criteria
        if criteria_type == "ALL":
            if criteria1 and criteria2:
                bar = frozenset(int(b) for b in bar)  # make int instead of np.int64
                new_barriers.append(bar)
        elif criteria_type == "JOSE":
            if criteria1 and criteria2 and criteria3 and criteria4:
                bar = frozenset(int(b) for b in bar)  # make int instead of np.int64
                new_barriers.append(bar)
        else:  # I choose to assume 'ANY' as the default
            if criteria1 or criteria2:
                bar = frozenset(int(b) for b in bar)  # make int instead of np.int64
                new_barriers.append(bar)

    return new_barriers


def get_best_next_barrier_set(df: pd.DataFrame, original_barriers) -> set:
    """
    Given the hex maze database and an original barrier set, find the best
    potential next barrier set (based on the number of hexes
    different on the optimal paths between reward ports).

    Parameters:
        df (pd.DataFrame): Database of potential hex maze configurations we want to search from
        original_barriers (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze to start from, represented in any valid format

    Returns:
        set: The "best" potential next barrier set (maximally different from the original barrier set)
    """
    # Convert all valid maze representations to a set of ints representing barrier hexes
    original_barriers = maze_to_barrier_set(original_barriers)

    # Get all potential next barrier sets (that differ from the original by the movement of a single barrier)
    potential_next_barriers = get_next_barrier_sets(df, original_barriers, criteria_type="ALL")

    # If there are no potential next barrier sets, return None
    if not potential_next_barriers:
        return None

    max_hex_diff = 0
    # Check how different each next barrier set is from our original barrier set
    for barriers in potential_next_barriers:
        hex_diff = num_hexes_different_on_optimal_paths(original_barriers, barriers)
        # If this barrier set is maximally different from the original, save it
        if hex_diff > max_hex_diff:
            max_hex_diff = hex_diff
            best_next_barriers = barriers

    return best_next_barriers


def find_all_valid_barrier_sequences(df: pd.DataFrame, start_barrier_set, min_hex_diff=8, max_sequence_length=5):
    """
    Finds all valid sequences of barriers starting from the given start_barrier_set.

    This function recursively generates all sequences of barrier sets where each barrier set
    in the sequence differs from the previous by the movement of a single barrier.
    The optimal paths that the rat can travel between reward ports must be different
    for all barrier sets in a sequence.

    Parameters:
        df (pd.DataFrame): Database of potential hex maze configurations we want to search from
        start_barrier_set (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze to start generating barrier sequences from, represented in any valid format
        min_hex_diff (int): The minimum combined number of hexes different between the most
            similar optimal paths between all 3 reward ports for all mazes in a sequence.
        max_sequence_length (int): The maximum length of a sequence to generate.

    Returns:
        list[list[set]]: A list of all valid sequences of barriers. Each sequence
            is represented as a list of barrier sets.
    """

    def helper(current_barrier_set: set, visited: set, current_length: int) -> list[list[set]]:
        """
        A helper function to recursively find all valid sequences of barrier sets.
        The "visited" set ensures that no barrier set is revisited to avoid cycles/repetitions.

        Parameters:
            current_barrier_set (set): The current barrier set being processed.
            visited (set): A set of barrier sets that have already been visited to avoid cycles.
            current_length (int): The current length of our generated sequence.

        Returns:
            list[list[set]]: A list of all valid barrier sequences starting from the
                current_barrier_set. Each sequence is represented as a list of barrier sets.
        """
        # print(f"Current set: {current_barrier_set}")
        # print(f"Visited: {visited}")

        # Base case: if we have reached the max sequence length, return the current barrier set
        if current_length >= max_sequence_length:
            return [[current_barrier_set]]

        # Search the database for all valid new barrier sets from the current barrier set
        next_sets = get_next_barrier_sets(df, current_barrier_set, criteria_type="ANY")

        # Remove the current barrier set from the next sets to avoid self-referencing
        next_sets = [s for s in next_sets if s != current_barrier_set]

        # Remove barrier sets that have the same optimal paths as any set in the sequence
        # ( Currently commenting this out because the hex difference criteria below is stronger! )
        # next_sets = [s for s in next_sets if not any(have_common_optimal_paths(df, s, v) for v in visited)]

        # Remove barrier sets with optimal paths too similar to any other barrier set in the sequence
        next_sets = [
            s for s in next_sets if all(num_hexes_different_on_optimal_paths(s, v) >= min_hex_diff for v in visited)
        ]

        # Initialize a list to store sequences
        sequences = []

        # Iterate over each next valid set
        for next_set in next_sets:
            if next_set not in visited:
                # Mark the next set as visited
                visited.add(next_set)

                # Recursively find sequences from the next set
                subsequences = helper(next_set, visited, current_length + 1)

                # Append the current set to the beginning of each subsequence
                for subsequence in subsequences:
                    sequences.append([current_barrier_set] + subsequence)

                # Unmark the next set as visited (backtrack)
                visited.remove(next_set)

        # If no valid sequences were found, return the current barrier set as the only sequence
        if not sequences:
            return [[current_barrier_set]]

        return sequences

    # Convert all valid maze representations to a set of ints representing barrier hexes
    start_barrier_set = maze_to_barrier_set(start_barrier_set)

    # Start the recursive search from the initial barrier set
    return helper(start_barrier_set, {frozenset(start_barrier_set)}, 1)


def get_barrier_sequence(
    df: pd.DataFrame,
    start_barrier_set,
    min_hex_diff=8,
    max_sequence_length=5,
    max_recursive_calls=40,
    criteria_type="ANY",
) -> list[set]:
    """
    Finds a sequence of barriers starting from the given start_barrier_set. This is a
    reasonably fast way to generate a good barrier sequence given a starting sequence
    (and is almost always preferable to generating all possible sequences using
    find_all_valid_barrier_sequences, which can take a very long time).

    This function recursively generates sequences of barrier sets where each barrier set
    in the sequence differs from the previous by the movement of a single barrier.
    The optimal paths that the rat can travel between reward ports must differ by at least
    min_hex_diff (default 8) hexes for all barrier sets in a sequence. This function may
    not return the best possible barrier sequence, but it wil return the longest valid
    barrier sequence found (up to max_sequence_length, default 5).

    Parameters:
        df (pd.DataFrame): Database of potential hex maze configurations we want to search from
        start_barrier_set (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze to start generating barrier sequences from, represented in any valid format
        min_hex_diff (int): The minimum combined number of hexes different between the most
            similar optimal paths between all 3 reward ports for all mazes in a sequence (default=8).
        max_sequence_length (int): The maximum length of a sequence to generate. Will stop
            searching and automatically return a sequence once we find one of this length (default=5).
        max_recursive_calls (int): The maximum number of recursive calls to make on our search
            for a good barrier sequence (so the function doesn't run for a really long time). Stops
            the search and returns the longest valid barrier sequence found by this point (default=40).
        criteria_type (str): The criteria type for what makes a valid next barrier set to propagate
            to get_next_barrier_sets. Options are 'ALL', 'ANY', or 'JOSE'. Defaults to 'ANY'

    Returns:
        list[set]: A valid sequence of barrier sets that is "good enough" (meaning it
            fulfills all our criteria but is not necessarily the best one), or the starting barrier
            set if no such sequence is found.
    """

    # Keep track of our longest sequence found in case we don't find one of max_sequence_length
    longest_sequence_found = []

    # Stop looking and return the longest sequence found after max_recursive_calls (for speed)
    recursive_calls = 0

    def helper(current_sequence: list[set], visited: set, current_length: int) -> list[set]:
        """
        A helper function to recursively find a "good enough" sequence of barrier sets.
        The "visited" set ensures that no barrier set is revisited to avoid cycles/repetitions.

        Parameters:
            current_sequence (list[set]): The current sequence of barrier sets being processed.
            visited (set): A set of barrier sets that have already been visited to avoid cycles.
            current_length (int): The current length of the sequence.

        Returns:
            list[set]: A valid sequence of barrier sets that is "good enough" (meaning it
                fulfills all our criteria but is not necessarily the best one), or the current
                barrier sequence if no such sequence is found.
        """
        # We keep track of these outside of the helper function
        nonlocal longest_sequence_found, recursive_calls

        # Keep track of how many times we have called the helper function
        recursive_calls += 1

        # print("in helper")
        # Base case: if the sequence length has reached the maximum, return the current sequence
        if current_length >= max_sequence_length:
            return current_sequence

        # print(f"Current sequence: {current_sequence}")

        # If this sequence is longer than our longest sequence found, it is our new longest sequence
        if current_length > len(longest_sequence_found):
            # print("This is our new longest sequence!")
            longest_sequence_found = current_sequence

        # If we have reached the limit of how many times to call helper, return the longest sequence found
        if recursive_calls >= max_recursive_calls:
            # print("Max recursive calls reached!")
            return longest_sequence_found

        current_barrier_set = current_sequence[-1]

        # Search the database for all valid new barrier sets from the current barrier set
        next_sets = get_next_barrier_sets(df, current_barrier_set, criteria_type=criteria_type)

        # Remove the current barrier set from the next sets to avoid self-referencing
        next_sets = [s for s in next_sets if s != current_barrier_set]

        # Remove barrier sets with optimal paths too similar to any other barrier set in the sequence
        next_sets = [
            s for s in next_sets if all(num_hexes_different_on_optimal_paths(s, v) >= min_hex_diff for v in visited)
        ]

        # Iterate over each next valid set
        for next_set in next_sets:
            if next_set not in visited:
                # Mark the next set as visited
                visited.add(next_set)

                # Recursively find sequences from the next set
                result = helper(current_sequence + [next_set], visited, current_length + 1)

                # If a sequence of the maximum length is found, return it
                if result and len(result) == max_sequence_length:
                    return result

                # Unmark the next set as visited (backtrack)
                visited.remove(next_set)

        # If no valid sequences were found, return the current sequence
        # print(f"Sequence at return: {current_sequence}")
        return current_sequence

    # Convert all valid maze representations to a set of ints representing barrier hexes
    start_barrier_set = maze_to_barrier_set(start_barrier_set)

    # Start the recursive search from the initial barrier set
    barrier_sequence = helper([start_barrier_set], {frozenset(start_barrier_set)}, 1)

    # print(f"Barrier sequence: {barrier_sequence}")
    # print(f"Longest sequence: {longest_sequence_found}")

    # Return the longest sequence
    return longest_sequence_found if len(longest_sequence_found) > len(barrier_sequence) else barrier_sequence


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


############## Use the above functions to get all the info about a maze configuration ##############


def df_lookup(df: pd.DataFrame, barriers, attribute_name: str):
    """
    Use the hex maze database to look up a specified attribute of a of a hex maze configuration.

    Note:
        The function `get_maze_attributes` is usually more practical. This helper was
        originally created when I thought a database lookup would be faster than direct
        calculation, but performance differences are negligible (even for ~50,000 queries).
        This function is retained for optional use.

    Parameters:
        df (pd.DataFrame): Database of hex maze configurations we want to search from
        barriers (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format
        attribute_name (str): The maze attribute to look up in the df.
            Must exist as a column in the df

    Returns:
        The value of the attribute for this maze, or None if the maze isn't in the df
    """
    # Check if the attribute_name exists as a column in the DataFrame
    if attribute_name not in df.columns:
        raise ValueError(f"Column '{attribute_name}' does not exist in the DataFrame.")

    # Convert all valid maze representations to a set of ints representing barrier hexes
    barriers = maze_to_barrier_set(barriers)

    # Filter the DataFrame
    filtered_df = df[df["barriers"] == barriers][attribute_name]

    # If this maze isn't found in the DataFrame, return None
    if filtered_df.empty:
        return None
    # Otherwise return the value of the attribute
    else:
        return filtered_df.iloc[0]


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


################################ Plotting hex mazes ################################


def get_distance_to_nearest_neighbor(hex_centroids: dict) -> dict:
    """
    Given a dictionary of hex: (x,y) centroid, calculate the minimum 
    euclidean distance to the closest neighboring hex centroid for each hex.

    Parameters:
        hex_centroids (dict): Dictionary of hex: (x, y) coords of centroid

    Returns:
        min_distances (dict): Dictionary of hex: minimum distance to the nearest hex
    """
    hex_ids = list(hex_centroids.keys())
    hex_coords = list(hex_centroids.values())

    # Use KDTree to find the closest hex
    tree = KDTree(hex_coords)

    # Query the nearest neighbor (k=2 because the closest hex centroid is itself)
    distances, _ = tree.query(hex_coords, k=2)

    # The first nearest is the hex itself (distance 0), so we take the second one
    min_distances = {
        hex_id: dist[1] for hex_id, dist in zip(hex_ids, distances)
    }
    return min_distances


def get_hex_sizes_from_centroids(hex_centroids: dict) -> dict:
    """
    Given a dictionary of hex: (x,y) centroid, calculate the height and radius
    (aka side length) of each hex, and the min/max/average hex height and radius.

    Parameters:
        hex_centroids (dict): Dictionary of hex: (x, y) coords of centroid

    Returns:
        dict: Dictionary containing 'hex_heights_dict', 'hex_radii_dict', 'avg_hex_height', 
            'max_hex_height', 'min_hex_height', 'avg_hex_radius', 'max_hex_radius', 'min_hex_radius'
    """
    # Get the minimum distance from each hex to its nearest neighbor (aka hex heights)
    hex_heights = get_distance_to_nearest_neighbor(hex_centroids)

    # Convert min distances (aka hex heights) to side lengths (radii)
    hex_radii = {
        hex_id: dist / math.sqrt(3) for hex_id, dist in hex_heights.items()
    }

    hex_sizes_dict = {
        'hex_heights_dict': hex_heights,
        'hex_radii_dict': hex_radii,
        'avg_hex_height': sum(hex_heights.values()) / len(hex_heights),
        'max_hex_height': max(hex_heights.values()),
        'min_hex_height': min(hex_heights.values()),
        'avg_hex_radius': sum(hex_radii.values()) / len(hex_radii),
        'max_hex_radius': max(hex_radii.values()),
        'min_hex_radius': min(hex_radii.values())
    }
    return hex_sizes_dict


def get_min_max_centroids(hex_centroids: dict) -> tuple[float, float, float, float]:
    """
    Given a dictionary of hex: (x, y) centroid, return the min and max
    values for x and y hex centroids. Helper for plotting

    Parameters:
        hex_centroids (dict): Dictionary of hex: (x, y) coords of centroid

    Returns:
        tuple: (min_x, max_x, min_y, max_y)
    """
    x_coords, y_coords = zip(*hex_centroids.values())
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)


def get_hex_centroids(view_angle=1, scale=1, shift=[0, 0]) -> dict:
    """
    Calculate the (x,y) coordinates of each hex centroid.
    Centroids are calculated relative to the centroid of the topmost hex at (0,0).

    Parameters:
        view_angle (int: 1, 2, or 3): The hex that is on the top point of the triangle
            when viewing the hex maze. Defaults to 1
        scale (int): The width of each hex (aka the length of the long diagonal,
            aka 2x the length of a single side). Defaults to 1
        shift (list): The x shift and y shift of the coordinates (after scaling),
            such that the topmost hex sits at (x_shift, y_shift) instead of (0,0).

    Returns:
        dict: a dictionary of hex: (x,y) coordinate of centroid
    """

    # Number of hexes in each vertical row of the hex maze
    hexes_per_row = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    # List of hexes in order from top to bottom, left to right (assuming view_angle=1)
    hex_list = [1, 4, 6, 5, 8, 7, 11, 10, 9, 14, 13, 12, 18, 17, 16, 15,
                22, 21, 20, 19, 27, 26, 25, 24, 23, 32, 31, 30, 29, 28,
                38, 37, 36, 35, 34, 33, 49, 42, 41, 40, 39, 48, 2, 47, 46, 45, 44, 43, 3]
    # If hex 2 should be on top instead, rotate hexes counterclockwise
    if view_angle == 2:
        hex_list = [rotate_hex(hex, direction="counterclockwise") for hex in hex_list]
    # If hex 3 should be on top instead, rotate hexes clockwise
    elif view_angle == 3:
        hex_list = [rotate_hex(hex, direction="clockwise") for hex in hex_list]

    # Vertical distance between rows for touching hexes
    y_offset = math.sqrt(3) / 2 * scale
    y_shift = 0
    count = 0
    hex_positions = {}
    for row, hexes_in_this_row in enumerate(hexes_per_row):
        if row % 2 == 0 and row != 0:  # Every other row, shift an extra 1/2 hex down
            y_shift += y_offset / 2
        for hex in range(hexes_in_this_row):
            x = hex * 3 / 2 * scale - (hexes_in_this_row - 1) * 3 / 4 * scale
            y = -row * y_offset + y_shift
            hex_positions[hex_list[count]] = (x, y)
            count += 1

    # Shift the coordinates by [x_shift, y_shift]
    x_shift = shift[0]
    y_shift = shift[1]
    hex_positions = {hex: (x + x_shift, y + y_shift) for hex, (x, y) in hex_positions.items()}

    return hex_positions


def classify_triangle_vertices(vertices: list[tuple]) -> dict:
    """
    Given a list of 3 triangle vertices, classify them as 'left', 'right' and
    'top' or 'bottom'. Useful for adjusting precise coordinates of where to plot
    stats on the maze graph so everything is beautiful and perfect.

    Parameters:
        vertices (list[tuple]): [(x1, y1), (x2, y2), (x3, y3)]

    Returns:
        dict: Dictionary of label: point, where label is left/right/top/bottom
    """
    left, mid, right = sorted(vertices, key=lambda p: p[0])
    avg_y = (left[1] + right[1]) / 2
    label = 'top' if mid[1] < avg_y else 'bottom'
    return {'left': left, 'right': right, label: mid}


def scale_triangle_from_centroid(vertices: list[tuple], shift: float) -> list[tuple]:
    """
    Shift triangle vertices outward or inward from the triangle centroid.
    Helper for plotting.

    Parameters:
        vertices (list[tuple]): [(x1, y1), (x2, y2), (x3, y3)]
        shift (float): Amount to shift vertices in (if negative) or out (if positive)

    Returns:
        list[tuple]: New triangle vertices after scaling
    """
    centroid = np.mean(vertices, axis=0)
    return [
        tuple(np.array(v) + (np.array(v) - centroid) * shift / np.linalg.norm(np.array(v) - centroid))
        for v in vertices
    ]


def get_base_triangle_coords(
    hex_positions, scale=1, chop_vertices=True, chop_vertices_2=False, show_edge_barriers=True
) -> list:
    """
    Calculate the coordinates of the vertices of the base triangle that
    surrounds all of the hexes in the maze.
    Used as an easy way to show permanent barriers when plotting the hex maze.

    Parameters:
        hex_positions (dict): A dictionary of hex: (x,y) coordinates of centroids.
        scale (int): The width of each hex (aka the length of the long diagonal,
            aka 2x the length of a single side). Defaults to 1
        chop_vertices (bool): If the vertices of the triangle should be chopped off
            (because there are no permanent barriers behind the reward ports).
            If True, returns 6 coords of a chopped triangle instead of just 3.
            Defaults to True (assuming exclude_edge_barriers is False)
        chop_vertices_2 (bool): If the vertices of the triangle should be chopped off
            twice as far as chop_vertices (to show 4 edge barriers instead of 6 per side).
            If True, returns 6 coords of a chopped triangle instead of just 3.
            Defaults to False. If True, makes chop_vertices False (both cannot be True)
        show_edge_barriers (bool): If False, returns a smaller triangle
            to plot the maze base without showing the permanent edge barriers.
            Defaults to True (bc why would you show the permanent barriers but not edges??)

    Returns:
        list: A list of (x, y) tuples representing the vertices of the maze base
    """

    # Get flat-to-flat height of hex (aka distance from top centroid to top of the triangle)
    height = scale * math.sqrt(3) / 2

    # Calculate triangle verticles based on hex height and centroids of reward port hexes
    vertices = [hex_positions.get(1), hex_positions.get(2), hex_positions.get(3)]
    vertices = scale_triangle_from_centroid(vertices, height)

    # Optionally make the triangle smaller so when we plot it behind the hex maze,
    # the permanent edge barriers don't show up. (This is ugly, I don't recommend)
    if not show_edge_barriers:
        # Move each vertex inward to shrink the triangle
        small_triangle_vertices = scale_triangle_from_centroid(vertices, -1*height)
        return small_triangle_vertices

    # Both cannot be True.
    if chop_vertices_2:
        chop_vertices = False

    # Chop off the tips of the triangle because there aren't barriers by reward ports
    # Returns 6 coordinates defining a chopped triangle instead of 3
    # This is used for defining the base footprint of the hex maze
    if chop_vertices:
        # Calculate new vertices for chopped corners
        chopped_vertices = []
        for i in range(3):
            p1 = np.array(vertices[i])  # Current vertex
            p2 = np.array(vertices[(i + 1) % 3])  # Next vertex

            # Calculate the new vertex by moving back from the current vertex
            unit_direction_vector = (p2 - p1) / np.linalg.norm(p1 - p2)
            # (to show 4 edge barriers instead of 6 (per side), multiply scale*2 here)
            chop_vertex1 = p1 + unit_direction_vector * scale
            chop_vertex2 = p2 - unit_direction_vector * scale
            chopped_vertices.append(tuple(chop_vertex1))
            chopped_vertices.append(tuple(chop_vertex2))
        return chopped_vertices

    # Chop off the tips of the triangle because there aren't barriers by reward ports
    # Returns 6 coordinates defining a chopped triangle instead of 3
    # This is used for defining the footprint of the permanent barriers
    if chop_vertices_2:
        # Calculate new vertices for chopped corners
        chopped_vertices = []
        for i in range(3):
            p1 = np.array(vertices[i])  # Current vertex
            p2 = np.array(vertices[(i + 1) % 3])  # Next vertex

            # Calculate the new vertex by moving back from the current vertex
            unit_direction_vector = (p2 - p1) / np.linalg.norm(p1 - p2)
            # Chop amount shows 4 edge barriers instead of 6 (per side)
            # (Note: For ideal centroids, we should move these vertices by scale*2, but we use scale*2.1 
            # to handle imprecision in custom hex coordinates. The extra bit is invisible behind hexes anyway)
            chop_vertex1 = p1 + unit_direction_vector * scale * 2.1
            chop_vertex2 = p2 - unit_direction_vector * scale * 2.1
            chopped_vertices.append(tuple(chop_vertex1))
            chopped_vertices.append(tuple(chop_vertex2))
        return chopped_vertices

    # Return the 3 vertices of the maze base
    else:
        return vertices


def get_stats_coords(hex_centroids: dict) -> dict:
    """
    When plotting a hex maze with additional stats (such as path lengths), get the
    graph coordinates of where to display those stats based on the hex centroids.

    Parameters:
        hex_centroids (dict): Dictionary of hex_id: (x, y) centroid of that hex

    Returns:
        stats_coords (dict): Dictionary of stat_id: (x, y) coordinates of where to plot it.
            The stat_id should be the same as a key returned by `get_maze_attributes`
    """
    # Get coordinates of reward port hexes
    hex1, hex2, hex3 = hex_centroids.get(1), hex_centroids.get(2), hex_centroids.get(3)

    # Get average hex size for scaling
    hex_sizes_dict = get_hex_sizes_from_centroids(hex_centroids)
    average_hex_radius = hex_sizes_dict.get("avg_hex_radius")

    # The path lengths between ports should be shown halfway between the reward port hexes
    len12 = tuple((np.array(hex1) + np.array(hex2)) / 2)
    len13 = tuple((np.array(hex1) + np.array(hex3)) / 2)
    len23 = tuple((np.array(hex2) + np.array(hex3)) / 2)
    # Move the coordinates outwards by 2.5 hex radii so they are outside the maze
    len12, len13, len23 = scale_triangle_from_centroid(vertices=[len12, len13, len23], shift=average_hex_radius*2.5)

    # The reward probabilities for each port should be shown outside the reward ports
    pA, pB, pC = scale_triangle_from_centroid(vertices=[hex1, hex2, hex3], shift=average_hex_radius*2.5)

    # Set up dict of stats coords
    stats_coords = {"len12": len12, "len13": len13, "len23": len23, "pA": pA, "pB": pB, "pC": pC}
    return stats_coords


def plot_hex_maze(
    barriers=None,
    old_barrier:int=None,
    new_barrier:int=None,
    show_barriers:bool=True,
    show_choice_points:bool=True,
    show_optimal_paths:bool=False,
    show_arrow:bool=True,
    show_barrier_change:bool=True,
    show_hex_labels:bool=True,
    show_stats:bool=True,
    reward_probabilities:list=None,
    show_permanent_barriers:bool=False,
    show_edge_barriers:bool=True,
    centroids:dict=None,
    view_angle:int=1,
    highlight_hexes=None,
    highlight_colors=None,
    scale=1,
    shift=[0, 0],
    ax=None,
    invert_yaxis:bool=False,
):
    """
    Given a set of barriers specifying a hex maze, plot the maze
    in classic hex maze style.
    Open hexes are shown in light blue. By default, barriers are shown
    in black, and choice point(s) are shown in yellow.

    Option to specify old_barrier hex and new_barrier hex
    to indicate a barrier change configuration:
    The now-open hex where the barrier used to be is shown in pale red.
    The new barrier is shown in dark red. An arrow indicating the movement
    of the barrier from the old hex to the new hex is shown in pink.

    Parameters:
        barriers (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze represented in any valid format (traditionally a set of barrier locations).
            If no barriers or 'None' is specifed, plots an empty hex maze
        old_barrier (int): Optional. The hex where the barrier was in the previous maze
        new_barrier (int): Optional. The hex where the new barrier is in this maze
        ax (matplotlib.axes.Axes): Optional. The axis on which to plot the hex maze.
            When no axis (or None) is specified, the function creates a new figure and shows the plot.

        show_barriers (bool): If the barriers should be shown as black hexes and labeled.
            If False, only open hexes are shown. Defaults to True
        show_choice_points (bool): If the choice points should be shown in yellow.
            If False, the choice points are not indicated on the plot. Defaults to True
        show_optimal_paths (bool): Highlight the hexes on optimal paths between
            reward ports in light green. Defaults to False
        show_arrow (bool): Draw an arrow indicating barrier movement from the
            old_barrier hex to the new_barrier hex. Defaults to True if old_barrier and
            new_barrier are not None
        show_barrier_change (bool): Highlight the old_barrier and new_barrier hexes
            on the maze. Defaults to True if old_barrier and new_barrier are not None.
        show_hex_labels (bool): Show the number of each hex on the plot. Defaults to True
        show_stats (bool): Print maze stats (lengths of optimal paths between ports)
            on the graph. Defaults to True
        reward_probabilities (list): Reward probabilities in format [pA, pB, pC] to print
            next to the reward port hexes. Defaults to None
        show_permanent_barriers (bool): If the permanent barriers should be shown
            as black hexes. Includes edge barriers. Defaults to False
        show_edge_barriers (bool): Only an option if show_permanent_barriers=True.
            Gives the option to exclude edge barriers when showing permanent barriers.
            Defaults to True if show_permanent_barriers=True
        centroids (dict): Dictionary of hex_id: (x, y) centroid of that hex. Must include
            all 49 hexes. Note that if centroids are very non-uniform, plot options
            like show_permanent_barriers may not work as well. Works best when only open hexes
            are shown. Defaults to None
        view_angle (int: 1, 2, or 3): The hex that is on the top point of the triangle
            when viewing the hex maze, if centroids is not specified. Defaults to 1
        highlight_hexes (set[int] or list[set]): A set (or list[set]) of hexes to highlight.
            Takes precedence over other hex highlights (choice points, etc). Defaults to None.
        highlight_colors (string or list[string]): Color (or list[colors]) to highlight highlight_hexes.
            Each color in this list applies to the respective set of hexes in highlight_hexes.
            Defaults to 'darkorange' for a single group.
        invert_yaxis (bool): Invert the y axis. Often useful when specifying centroids based on
            video pixel coordinates, as video uses top left as (0,0), effectively vertically 
            flipping the hex maze when plotting the centroids on "normal" axes. Defaults to False

    Other function behavior to note:
    - If centroids argument is specified, view_angle will be ignored (centroid coordinates determine both
        hex scale and view angle)
    - Hexes specified in highlight_hexes takes precedence over all other highlights.
    - If the same hex is specified multiple times in highlight_hexes, the last time takes precedence.
    - Highlighting choice points takes precedence over highlighting barrier change hexes,
        as they are also shown by the movement arrow. If show_barriers=False, the new_barrier hex
        will not be shown even if show_barrier_change=True (because no barriers are shown with this option.)
    - show_optimal_paths has the lowest precedence (will be overridden by all other highlights).
    """
    # Create an empty hex maze
    hex_maze = create_empty_hex_maze()

    # If the user specified a dictionary of hex centroids, use these 
    if centroids is not None:
        # Make a copy to avoid modifying the original centroids dict
        hex_coordinates = centroids.copy()
        hex_sizes_dict = get_hex_sizes_from_centroids(hex_coordinates)
        hex_radii_dict = hex_sizes_dict.get('hex_radii_dict')

        # Use the custom centroids to calculate scale (overrides user-specfied args)
        scale = hex_sizes_dict.get("avg_hex_radius")*2
    else:
        # Otherwise, get a dictionary of the (x,y) coordinates of each hex centroid based on maze view angle
        hex_coordinates = get_hex_centroids(view_angle=view_angle, scale=scale, shift=shift)

    # Get a dictionary of stats coordinates based on hex coordinates
    if show_stats or reward_probabilities is not None:
        stats_coordinates = get_stats_coords(hex_coordinates)
    # Define this for times we want to draw the arrow but not show barriers
    new_barrier_coords = None

    # Make the open hexes light blue
    hex_colors = {node: "skyblue" for node in hex_maze.nodes()}

    if barriers is not None:
        # Convert all valid maze representations to a set of ints representing barrier hexes
        barriers = maze_to_barrier_set(barriers)

        # Make the barriers black if we want to show them
        if show_barriers:
            for hex in barriers:
                hex_colors.update({hex: "black"})
        # Or if we don't want to show the barriers, remove them
        else:
            # Save the coordinates of the new barrier hex if we still want to draw the arrow
            if new_barrier is not None and show_arrow:
                new_barrier_coords = hex_coordinates.pop(new_barrier, None)
            # Remove the barrier hexes from all of our dicts
            for hex in barriers:
                hex_coordinates.pop(hex, None)
                hex_colors.pop(hex, None)
                hex_maze.remove_node(hex)

        # Optional - Make the hexes on optimal paths light green
        if show_optimal_paths:
            hexes_on_optimal_paths = {hex for path in get_optimal_paths_between_ports(barriers) for hex in path}
            for hex in hexes_on_optimal_paths:
                hex_colors.update({hex: "lightgreen"})

        # Optional - Make the old barrier location (now an open hex) light red to indicate barrier change
        if old_barrier is not None and show_barrier_change:
            hex_colors.update({old_barrier: "peachpuff"})

        # Optional - Make the new barrier location dark red to indicate barrier change
        if new_barrier is not None and show_barrier_change:
            hex_colors.update({new_barrier: "darkred"})

        # Optional - Make the choice point(s) yellow
        if show_choice_points:
            choice_points = get_critical_choice_points(barriers)
            for hex in choice_points:
                hex_colors.update({hex: "gold"})

    else:
        # If barriers = None, there are no real stats to show
        show_stats = False

    # Optional - highlight specific hexes on the plot
    if highlight_hexes is not None:
        # If highlight_hexes is a single set (or a list of length 1 containing a set),
        # default to dark orange if no colors are provided
        if isinstance(highlight_hexes, set) or (
            isinstance(highlight_hexes, list) and len(highlight_hexes) == 1 and isinstance(highlight_hexes[0], set)
        ):
            if highlight_colors is None:
                highlight_colors = ["darkorange"]

            # If it's a single set, wrap it in a list for consistency
            if isinstance(highlight_hexes, set):
                highlight_hexes = [highlight_hexes]

        # If highlight_hexes is a list, ensure highlight_colors is the same length
        # (We actually just check if len(colors) is >= len(hexes) and ignore any extra colors)
        elif isinstance(highlight_hexes, list):
            if highlight_colors is None or len(highlight_hexes) > len(highlight_colors):
                raise ValueError("Length of highlight_colors and highlight_hexes must match.")

        # Apply the specified or default color to the hexes
        for hexes, color in zip(highlight_hexes, highlight_colors):
            for hex in hexes:
                hex_colors.update({hex: color})

    # If no axis was provided, create a new figure and axis to use
    if ax is None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        show_plot = False

    # Show permanent barriers by adding a barrier-colored background before plotting the maze
    if show_barriers and show_permanent_barriers:
        # Add a big triangle in light blue to color the open half-hexes next to reward ports
        base_vertices1 = get_base_triangle_coords(
            hex_positions=hex_coordinates, scale=scale, show_edge_barriers=show_edge_barriers
        )
        maze_base1 = patches.Polygon(base_vertices1, closed=True, facecolor="skyblue", fill=True)
        ax.add_patch(maze_base1)
        # Add a big triangle with the edges cut off in black to color the other half-hexes on the side as barriers
        base_vertices2 = get_base_triangle_coords(
            hex_positions=hex_coordinates, scale=scale, show_edge_barriers=show_edge_barriers, chop_vertices_2=True
        )
        maze_base2 = patches.Polygon(base_vertices2, closed=True, facecolor="black", fill=True)
        ax.add_patch(maze_base2)

    # Add each hex to the plot
    for hex, (x, y) in hex_coordinates.items():
        hexagon = patches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=scale / 2 if centroids is None else hex_radii_dict.get(hex),
            orientation=math.pi / 6,
            facecolor=hex_colors[hex],
            edgecolor="white",
        )
        ax.add_patch(hexagon)

    # If we have a barrier change, add an arrow between the old_barrier and new_barrier to show barrier movement
    if show_arrow and old_barrier is not None and new_barrier is not None:
        arrow_start = hex_coordinates[old_barrier]
        # If we removed new_barrier from our dict (because show_barriers=False), use the saved coordinates
        arrow_end = hex_coordinates.get(new_barrier, new_barrier_coords)
        ax.annotate(
            "",
            xy=arrow_end,
            xycoords="data",
            xytext=arrow_start,
            textcoords="data",
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3, rad=0.2", color="salmon", linewidth=2),
        )

    # Add hex labels
    if show_hex_labels:
        nx.draw_networkx_labels(
            hex_maze, hex_coordinates, labels={h: h for h in hex_maze.nodes()}, font_color="black", ax=ax
        )

    # Add barrier labels
    if barriers is not None and show_barriers and show_hex_labels:
        nx.draw_networkx_labels(hex_maze, hex_coordinates, labels={b: b for b in barriers}, font_color="white", ax=ax)

    # Optional - Add stats to the graph
    if show_stats and barriers is not None:
        # Get stats for this maze
        maze_attributes = get_maze_attributes(barriers)
        # For all stats that we have display coordinates for, print them on the graph
        # (Currently this is just optimal path lengths, but it would be easy to add others!)
        for stat in maze_attributes:
            if stat in stats_coordinates:
                ax.annotate(maze_attributes[stat], stats_coordinates[stat], ha="center", fontsize=12)

    # Optional - Add reward probabilites to the graph
    if reward_probabilities is not None:
        for name, prob in zip(["pA", "pB", "pC"], reward_probabilities):
            ax.annotate(f"{prob}%", stats_coordinates[name], ha="center", fontsize=12)

    # Classify reward port hexes as left, right, and bottom or top 
    # so we know if we need to add space for stats on the top vs the bottom of the maze
    labeled_vertices = classify_triangle_vertices([hex_coordinates[1], hex_coordinates[2], hex_coordinates[3]])

    # Set default axes limits
    min_x, max_x, min_y, max_y = get_min_max_centroids(hex_coordinates)
    xlim = [min_x - scale, max_x + scale]
    ylim = [min_y - scale, max_y + scale]

    # If showing reward probabilites, add a little space 
    if reward_probabilities is not None:
        xlim = [xlim[0] - scale, xlim[1] + scale]
        ylim = [ylim[0] - scale, ylim[1] + scale]

    # If showing path length stats, add a little space 
    if show_stats and (reward_probabilities is None):
        if "top" in labeled_vertices:
            ylim[1] += scale # add space on bottom (shift view upward)
        else:
            ylim[0] -= scale # add space on top (shift view downward)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")

    # Optional - invert yaxis. 
    # This can be useful when plotting a hex maze with hex centroids from the maze video,
    # as video pixel coordinates use top left as (0,0) so the maze appears flipped when
    # hexes in video coordinates are plotted on "normal" axes
    if invert_yaxis:
        ax.invert_yaxis()

    # If no axis was provided as an argument, show the plot now
    if show_plot:
        plt.show()


def plot_barrier_change_sequence(barrier_sequence: list[set], print_barrier_info=True, same_plot=False, **kwargs):
    """
    Given a sequence of barrier sets that each differ by the movement of
    a single barrier, plot each maze in the sequence with the moved barriers
    indicated on each maze.

    Open hexes are shown in light blue. By default, barriers are shown
    in black, and choice point(s) are shown in yellow.
    The now-open hex where the barrier used to be is shown in pale red.
    The new barrier is shown in dark red. An arrow indicating the movement
    of the barrier from the old hex to the new hex is shown in pink.

    Parameters:
        barrier_sequence (list[set]): List of sequential barrier sets
        print_barrier_info (bool): Optional. Print each barrier set and the
            barrier moved between barrier sets. Defaults to True
        same_plot (bool). Optional. Prints all mazes in a single row as
            subplots in the same plot (instead of as separate figures).
            Defaults to False

        show_barriers (bool): If the barriers should be shown as black hexes and labeled.
            If False, only open hexes are shown. Defaults to True
        show_choice_points (bool): If the choice points should be shown in yellow.
            If False, the choice points are not indicated on the plot. Defaults to True
        show_optimal_paths (bool): Highlight the hexes on optimal paths between
            reward ports in light green. Defaults to False
        show_arrow (bool): Draw an arrow indicating barrier movement from the
            old_barrier hex to the new_barrier hex. Defaults to True if old_barrier and
            new_barrier are not None
        show_barrier_change (bool): Highlight the old_barrier and new_barrier hexes
            on the maze. Defaults to True if old_barrier and new_barrier are not None.
        show_hex_labels (bool): Show the number of each hex on the plot. Defaults to True
        show_stats (bool): Print maze stats (lengths of optimal paths between ports)
            on the graph. Defaults to False
        show_permanent_barriers (bool): If the permanent barriers should be shown
            as black hexes. Includes edge barriers. Defaults to False
        show_edge_barriers (bool): Only an option if show_permanent_barriers=True.
            Gives the option to exclude edge barriers when showing permanent barriers.
            Defaults to True if show_permanent_barriers=True
        view_angle (int: 1, 2, or 3): The hex that is on the top point of the triangle
            when viewing the hex maze. Defaults to 1
        highlight_hexes (set[int] or list[set]): A set (or list[set]) of hexes to highlight.
            Takes precedence over other hex highlights (choice points, etc). Defaults to None.
        highlight_colors (string or list[string]): Color (or list[colors]) to highlight highlight_hexes.
            Each color in this list applies to the respective set of hexes in highlight_hexes.
            Defaults to 'darkorange' for a single group.

    Other function behavior to note:
    - Hexes specified in highlight_hexes takes precedence over all other highlights.
    - If the same hex is specified multiple times in highlight_hexes, the last time takes precedence.
    - Highlighting choice points takes precedence over highlighting barrier change hexes,
        as they are also shown by the movement arrow. If show_barriers=False, the new_barrier hex
        will not be shown even if show_barrier_change=True (because no barriers are shown with this option.)
    - show_optimal_paths has the lowest precedence (will be overridden by all other highlights).
    """

    # Find the barriers moved from one configuration to the next
    barrier_changes = get_barrier_changes(barrier_sequence)

    # If we want all mazes in a row on the same plot, use this
    if same_plot:
        # Set up a 1x(num mazes) plot so we can put each maze in a subplot
        fig, axs = plt.subplots(1, len(barrier_sequence), figsize=((len(barrier_sequence) * 4, 4)))

        # Plot the first maze in the sequence (no barrier changes to show for Maze 1)
        plot_hex_maze(barrier_sequence[0], ax=axs[0], **kwargs)
        axs[0].set_title(f"Maze 1")

        # Loop through each successive maze in the sequence and plot it with barrier changes
        for i, (maze, (old_barrier_hex, new_barrier_hex)) in enumerate(
            zip(barrier_sequence[1:], barrier_changes), start=1
        ):
            # Plot the hex maze (pass any additional style args directly to plot_hex_maze)
            plot_hex_maze(maze, ax=axs[i], old_barrier=old_barrier_hex, new_barrier=new_barrier_hex, **kwargs)
            axs[i].set_title(f"Maze {i+1}")
            if print_barrier_info:
                axs[i].set_xlabel(f"Barrier change: {old_barrier_hex} → {new_barrier_hex}")

        # Adjust layout to ensure plots don't overlap
        plt.tight_layout()
        plt.show()

    # Otherwise, plot each maze separately
    else:
        # First print info for and plot the first maze (no barrier changes for Maze 1)
        if print_barrier_info:
            print(f"Maze 0: {barrier_sequence[0]}")
        plot_hex_maze(barrier_sequence[0], **kwargs)

        # Now plot each successive maze (and print barrier change info)
        for i, (barriers, (old_barrier_hex, new_barrier_hex)) in enumerate(zip(barrier_sequence[1:], barrier_changes)):
            if print_barrier_info:
                print(f"Barrier change: {old_barrier_hex} -> {new_barrier_hex}")
                print(f"Maze {i+1}: {barriers}")
            plot_hex_maze(barriers, old_barrier=old_barrier_hex, new_barrier=new_barrier_hex, **kwargs)


def plot_hex_maze_comparison(maze_1, maze_2, print_info=True, **kwargs):
    """
    Given 2 hex mazes, plot each maze highlighting the different hexes the
    rat must run through on optimal paths between reward ports. Used for comparing
    how different 2 mazes are.

    Open hexes are shown in light blue. By default, barriers are not shown.
    Changes in optimal paths between the mazes are highlighted in orange.

    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph):
            The second hex maze represented in any valid format
        print_info (bool): Optional. Print the hexes different on optimal paths between the mazes.
            Defaults to True

        show_barriers (bool): If the barriers should be shown as black hexes and labeled.
            If False, only open hexes are shown. Defaults to False
        show_choice_points (bool): If the choice points should be shown in yellow.
            If False, the choice points are not indicated on the plot. Defaults to True
        show_optimal_paths (bool): Highlight the hexes on optimal paths between
            reward ports in light green. Defaults to False
        show_arrow (bool): Draw an arrow indicating barrier movement from the
            old_barrier hex to the new_barrier hex. Defaults to True if old_barrier and
            new_barrier are not None
        show_barrier_change (bool): Highlight the old_barrier and new_barrier hexes
            on the maze. Defaults to True if old_barrier and new_barrier are not None.
        show_hex_labels (bool): Show the number of each hex on the plot. Defaults to True
        show_stats (bool): Print maze stats (lengths of optimal paths between ports)
            on the graph. Defaults to True
        show_permanent_barriers (bool): If the permanent barriers should be shown
            as black hexes. Includes edge barriers. Defaults to False
        show_edge_barriers (bool): Only an option if show_permanent_barriers=True.
            Gives the option to exclude edge barriers when showing permanent barriers.
            Defaults to True if show_permanent_barriers=True
        view_angle (int: 1, 2, or 3): The hex that is on the top point of the triangle
            when viewing the hex maze. Defaults to 1
        highlight_hexes (set[int] or list[set]): A set (or list[set]) of hexes to highlight.
            Takes precedence over other hex highlights (choice points, etc). Defaults to None.
        highlight_colors (string or list[string]): Color (or list[colors]) to highlight highlight_hexes.
            Each color in this list applies to the respective set of hexes in highlight_hexes.
            Defaults to 'darkorange' for a single group.

    Other function behavior to note:
    - Hexes specified in highlight_hexes takes precedence over all other highlights.
    - If the same hex is specified multiple times in highlight_hexes, the last time takes precedence.
    - Highlighting choice points takes precedence over highlighting barrier change hexes,
        as they are also shown by the movement arrow. If show_barriers=False, the new_barrier hex
        will not be shown even if show_barrier_change=True (because no barriers are shown with this option.)
    - show_optimal_paths has the lowest precedence (will be overridden by all other highlights).
    """

    # Get the hexes different on optimal paths between these 2 mazes
    hexes_maze1_not_maze2, hexes_maze2_not_maze1 = hexes_different_on_optimal_paths(maze_1, maze_2)

    # Print the different hexes
    if print_info:
        print(f"Hexes on optimal paths in the first maze but not the second: {hexes_maze1_not_maze2}")
        print(f"Hexes on optimal paths in the second maze but not the first: {hexes_maze2_not_maze1}")

        hex_diff = num_hexes_different_on_optimal_paths(maze_1, maze_2)
        print(f"There are {hex_diff} hexes different on optimal paths between the 2 mazes.")

    # By default, make show_stats=True and show_barriers=False
    kwargs.setdefault("show_barriers", False)
    kwargs.setdefault("show_stats", True)

    # Plot the mazes in side-by-side subplots highlighting different hexes
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plot_hex_maze(maze_1, ax=axs[0], highlight_hexes=hexes_maze1_not_maze2, **kwargs)
    plot_hex_maze(maze_2, ax=axs[1], highlight_hexes=hexes_maze2_not_maze1, **kwargs)
    axs[0].set_title("Maze 1")
    axs[1].set_title("Maze 2")
    plt.tight_layout()
    plt.show()


def plot_hex_maze_path_comparison(maze_1, maze_2, print_info=True, **kwargs):
    """
    Given 2 hex mazes, plot each maze highlighting the different hexes the 
    rat must run through on optimal paths between reward ports. Creates a 2x3
    plot, with each column highlighting the differences in paths between each
    pair of reward ports. Used for comparing how different 2 mazes are.
    
    Open hexes are shown in light blue. By default, barriers are not shown.
    Optimal paths between ports are highlighted in light green.
    Changes in optimal paths between the mazes are highlighted in orange.
    
    Parameters:
        maze_1 (list, set, frozenset, np.ndarray, str, nx.Graph): 
            The first hex maze represented in any valid format
        maze_2 (list, set, frozenset, np.ndarray, str, nx.Graph): 
            The second hex maze represented in any valid format
        print_info (bool): Optional. Print the hexes different on optimal paths between the mazes.
            Defaults to True

        show_barriers (bool): If the barriers should be shown as black hexes and labeled.
            If False, only open hexes are shown. Defaults to False
        show_choice_points (bool): If the choice points should be shown in yellow.
            If False, the choice points are not indicated on the plot. Defaults to False
        show_optimal_paths (bool): Highlight the hexes on optimal paths between
            reward ports in light green. Defaults to False
        show_arrow (bool): Draw an arrow indicating barrier movement from the
            old_barrier hex to the new_barrier hex. Defaults to True if old_barrier and
            new_barrier are not None
        show_barrier_change (bool): Highlight the old_barrier and new_barrier hexes
            on the maze. Defaults to True if old_barrier and new_barrier are not None.
        show_hex_labels (bool): Show the number of each hex on the plot. Defaults to True
        show_stats (bool): Print maze stats (lengths of optimal paths between ports)
            on the graph. Defaults to True
        show_permanent_barriers (bool): If the permanent barriers should be shown
            as black hexes. Includes edge barriers. Defaults to False
        show_edge_barriers (bool): Only an option if show_permanent_barriers=True.
            Gives the option to exclude edge barriers when showing permanent barriers.
            Defaults to True if show_permanent_barriers=True
        view_angle (int: 1, 2, or 3): The hex that is on the top point of the triangle
            when viewing the hex maze. Defaults to 1

    Note that this function passes the arguments highlight_hexes and highlight_colors \
    directly to plot_hex_maze to show differences in optimal paths. \
    Setting these arguments will render this function pointless, so don't!! 
    """

    # Get which hexes are different on the most similar optimal paths from port 1 to port 2
    maze1_optimal_12 = get_optimal_paths(maze_1, start_hex=1, target_hex=2)
    maze2_optimal_12 = get_optimal_paths(maze_2, start_hex=1, target_hex=2)
    maze1_hexes_path12, maze2_hexes_path12 = hexes_different_between_paths(maze1_optimal_12, maze2_optimal_12)
    num_hexes_different_path12 = len(maze1_hexes_path12 | maze2_hexes_path12)
    # Get which hexes are different on the most similar optimal paths from port 1 to port 3
    maze1_optimal_13 = get_optimal_paths(maze_1, start_hex=1, target_hex=3)
    maze2_optimal_13 = get_optimal_paths(maze_2, start_hex=1, target_hex=3)
    maze1_hexes_path13, maze2_hexes_path13 = hexes_different_between_paths(maze1_optimal_13, maze2_optimal_13)
    num_hexes_different_path13 = len(maze1_hexes_path13 | maze2_hexes_path13)
    # Get which hexes are different on the most similar optimal paths from port 2 to port 3
    maze1_optimal_23 = get_optimal_paths(maze_1, start_hex=2, target_hex=3)
    maze2_optimal_23 = get_optimal_paths(maze_2, start_hex=2, target_hex=3)
    maze1_hexes_path23, maze2_hexes_path23 = hexes_different_between_paths(maze1_optimal_23, maze2_optimal_23)
    num_hexes_different_path23 = len(maze1_hexes_path23 | maze2_hexes_path23)

    # By default, make show_stats=True, show_barriers=False, show_choice_points=False
    kwargs.setdefault("show_barriers", False)
    kwargs.setdefault("show_choice_points", False)
    kwargs.setdefault("show_stats", True)

    # Plot the mazes in side-by-side subplots highlighting different hexes
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    plot_hex_maze(
        maze_1,
        ax=axs[0, 0],
        highlight_hexes=[set(chain.from_iterable(maze1_optimal_12)), maze1_hexes_path12],
        highlight_colors=["lightgreen", "darkorange"],
        **kwargs,
    )
    plot_hex_maze(
        maze_2,
        ax=axs[1, 0],
        highlight_hexes=[set(chain.from_iterable(maze2_optimal_12)), maze2_hexes_path12],
        highlight_colors=["lightgreen", "darkorange"],
        **kwargs,
    )
    plot_hex_maze(
        maze_1,
        ax=axs[0, 1],
        highlight_hexes=[set(chain.from_iterable(maze1_optimal_13)), maze1_hexes_path13],
        highlight_colors=["lightgreen", "darkorange"],
        **kwargs,
    )
    plot_hex_maze(
        maze_2,
        ax=axs[1, 1],
        highlight_hexes=[set(chain.from_iterable(maze2_optimal_13)), maze2_hexes_path13],
        highlight_colors=["lightgreen", "darkorange"],
        **kwargs,
    )
    plot_hex_maze(
        maze_1,
        ax=axs[0, 2],
        highlight_hexes=[set(chain.from_iterable(maze1_optimal_23)), maze1_hexes_path23],
        highlight_colors=["lightgreen", "darkorange"],
        **kwargs,
    )
    plot_hex_maze(
        maze_2,
        ax=axs[1, 2],
        highlight_hexes=[set(chain.from_iterable(maze2_optimal_23)), maze2_hexes_path23],
        highlight_colors=["lightgreen", "darkorange"],
        **kwargs,
    )
    axs[0, 0].set_ylabel("Maze 1")
    axs[1, 0].set_ylabel("Maze 2")
    axs[0, 0].set_title(f"Hexes different between port 1 and 2")
    axs[1, 0].set_xlabel(f"{num_hexes_different_path12} hexes different between port 1 and 2")
    axs[0, 1].set_title(f"Hexes different between port 1 and 3")
    axs[1, 1].set_xlabel(f"{num_hexes_different_path13} hexes different between port 1 and 3")
    axs[0, 2].set_title(f"Hexes different between port 2 and 3")
    axs[1, 2].set_xlabel(f"{num_hexes_different_path23} hexes different between port 2 and 3")
    plt.tight_layout()
    plt.show()

    # Print the different hexes
    if print_info:
        # Get the hexes different on optimal paths between these 2 mazes
        hexes_maze1_not_maze2, hexes_maze2_not_maze1 = hexes_different_on_optimal_paths(maze_1, maze_2)

        print(f"Hexes on optimal paths in maze 1 but not maze 2: {hexes_maze1_not_maze2}")
        print(f"Hexes on optimal paths in maze 2 but not maze 1: {hexes_maze2_not_maze1}")
        hex_diff = num_hexes_different_on_optimal_paths(maze_1, maze_2)
        print(f"There are {hex_diff} hexes different across all optimal paths (not double counting hexes).")


def plot_evaluate_maze_sequence(barrier_sequence: list[set], **kwargs):
    """
    Given a sequence of barrier sets that each differ by the movement of
    a single barrier, plot each maze in the sequence showing a comparison of
    how different it is from every other maze in the sequence.

    Open hexes are shown in light blue. By default, barriers are not shown.
    The reference maze has optimal paths highlighted in green. It is shown
    in a row compared to all other mazes in the sequence, where hexes on
    optimal paths in the other maze that are not on optimal paths in the
    reference maze are highlighted in orange.

    Parameters:
        barrier_sequence (list[set]): List of sequential barrier sets

        show_barriers (bool): If the barriers should be shown as black hexes and labeled.
            If False, only open hexes are shown. Defaults to False
        show_choice_points (bool): If the choice points should be shown in yellow.
            If False, the choice points are not indicated on the plot. Defaults to False
        show_optimal_paths (bool): Highlight the hexes on optimal paths between
            reward ports in light green. Defaults to False
        show_arrow (bool): Draw an arrow indicating barrier movement from the
            old_barrier hex to the new_barrier hex. Defaults to True if old_barrier and
            new_barrier are not None
        show_barrier_change (bool): Highlight the old_barrier and new_barrier hexes
            on the maze. Defaults to True if old_barrier and new_barrier are not None.
        show_hex_labels (bool): Show the number of each hex on the plot. Defaults to False
        show_stats (bool): Print maze stats (lengths of optimal paths between ports)
            on the graph. Defaults to True
        show_permanent_barriers (bool): If the permanent barriers should be shown
            as black hexes. Includes edge barriers. Defaults to False
        show_edge_barriers (bool): Only an option if show_permanent_barriers=True.
            Gives the option to exclude edge barriers when showing permanent barriers.
            Defaults to True if show_permanent_barriers=True
        view_angle (int: 1, 2, or 3): The hex that is on the top point of the triangle
            when viewing the hex maze. Defaults to 1
    """

    # Change some default plotting options for clarity
    kwargs.setdefault("show_barriers", False)
    kwargs.setdefault("show_stats", True)
    kwargs.setdefault("show_choice_points", False)
    kwargs.setdefault("show_hex_labels", False)

    # Loop through each maze in the sequence
    for ref, maze in enumerate(barrier_sequence):

        # Compare the maze with each other maze in the sequence
        fig, axs = plt.subplots(1, len(barrier_sequence), figsize=(18, 3))

        for i, other_maze in enumerate(barrier_sequence):
            if i == ref:
                # If this maze is the reference for this row, highlight optimal paths
                plot_hex_maze(maze, ax=axs[i], show_optimal_paths=True, **kwargs)
                axs[i].set_title(f"Maze {i+1}")
            else:
                # Otherwise, get the hexes different on optimal paths between the reference maze and another maze in the sequence
                _, optimal_hexes_other_maze_not_reference_maze = hexes_different_on_optimal_paths(maze, other_maze)

                # Plot the other maze highlighting the hexes different from the reference maze
                plot_hex_maze(
                    other_maze, ax=axs[i], highlight_hexes=optimal_hexes_other_maze_not_reference_maze, **kwargs
                )
                axs[i].set_title(f"Maze {i+1} compared to Maze {ref+1}")
                axs[i].set_xlabel(f"{len(optimal_hexes_other_maze_not_reference_maze)} hexes different")

        # Adjust layout to ensure plots don't overlap
        plt.tight_layout()
        plt.show()


############## One-time use functions to help ensure that our database includes all possible mazes ##############


def num_isomorphic_mazes_in_set(set_of_valid_mazes: set[frozenset], maze) -> tuple[int, list[set]]:
    """
    Given a set of all valid maze configurations and a single hex maze configuration,
    find all isomorphic mazes for this configuration that already exist in our larger set,
    and which are missing.

    Parameters:
        set_of_valid_mazes (set[frozenset]): Set of all valid maze configurations
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze to check, represented in any valid format

    Returns:
        tuple:
            int: The number of isomorphic mazes that already exist in the set
            list[set]: A list of isomorphic maze configurations missing from the set
    """
    # Get all potential isomorphic mazes for this barrier configuration
    all_isomorphic_barriers = get_isomorphic_mazes(maze)
    # Find other mazes in the set that are isomorphic to the given barrier set
    isomorphic_barriers_in_set = set([b for b in set_of_valid_mazes if b in all_isomorphic_barriers])
    # Get the isomorphic mazes not present in the set
    isomorphic_bariers_not_in_set = all_isomorphic_barriers.difference(isomorphic_barriers_in_set)
    return len(isomorphic_barriers_in_set), isomorphic_bariers_not_in_set


def num_isomorphic_mazes_in_df(df: pd.DataFrame, maze) -> tuple[int, list[set]]:
    """
    Given our maze configuration database and a set of barriers defining
    a hex maze configuration, find all isomorphic mazes that already exist
    in the DataFrame, and which are missing.

    Parameters:
        df (pd.DataFrame): Database of hex maze configurations we want to search from,
            where maze configurations are specified in the column 'barriers'
        maze (list, set, frozenset, np.ndarray, str, nx.Graph):
            The hex maze to check, represented in any valid format

    Returns:
        tuple:
            int: The number of isomorphic mazes that already exist in the DataFrame
            list[set]: A list of isomorphic maze configurations missing from the DataFrame
    """
    # Get all potential isomorphic mazes for this barrier configuration
    all_isomorphic_barriers = get_isomorphic_mazes(maze)
    # Find other mazes in the DataFrame that are isomorphic to the given barrier set
    isomorphic_barriers_in_df = set([b for b in df["barriers"] if b in all_isomorphic_barriers])
    # Get the isomorphic mazes not present in the DataFrame
    isomorphic_bariers_not_in_df = all_isomorphic_barriers.difference(isomorphic_barriers_in_df)
    return len(isomorphic_barriers_in_df), isomorphic_bariers_not_in_df
