"""
utils.py

This module contains basic utility functions for transforming different hex maze 
representations, as well as maze rotations and reflections.
"""

import networkx as nx
import numpy as np
import pandas as pd


# Define the public interface for this module
__all__ = [
    "set_to_string", 
    "string_to_set",
    "create_empty_hex_maze",
    "maze_to_graph",
    "maze_to_barrier_set",
    "maze_to_string",
    "rotate_hex",
    "reflect_hex",
    "get_rotated_barriers",
    "get_reflected_barriers",
    "get_isomorphic_mazes",
    "df_lookup",
]


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


############## Functions for maze rotations and reflections across its axis of symmetry ##############


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


# num_isomorphic_mazes_in_set and num_isomorphic_mazes_in_df are one-time use functions to ensure we have
# all possible mazes in our hex maze database. They are not included in the public interface.

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