import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import random
import math
from functools import partial

# for now this is defined here because we use it to set up constants
def get_subpaths(path, length):
    ''' Given a path, return a set of all sub-paths of the specified length. '''
    return {tuple(path[i:i + length]) for i in range(len(path) - length + 1)}


################################# Set up all of our constants #################################

# Barriers can exist in any open hex, excluding the hexes right next to the reward ports
POSSIBLE_BARRIERS = np.arange(5, 48)
POSSIBLE_BARRIERS = POSSIBLE_BARRIERS[~np.isin(POSSIBLE_BARRIERS, [5, 6, 33, 38, 43, 47])]

# Minimum distance from port to critical choice point = 6 (including port hex)
ILLEGAL_CHOICE_POINTS_6 = [4, 6, 5, 11, 8, 10, 7, 9, 49, 38, 47, 32, 42, 27, 37, 46, 48, 43, 33, 39, 28, 44, 34, 23]

# Max straight path length to reward port = 6 hexes. (illegal paths are 7+)
MAX_STRAIGHT_PATH_TO_PORT = 6
STRAIGHT_PATHS_TO_PORTS = [[1, 4, 6, 8, 11, 14, 18, 22, 27, 32, 38, 49, 2], 
                  [1, 4, 5, 7, 9, 12, 15, 19, 23, 28, 33, 48, 3],
                  [2, 49, 47, 42, 46, 41, 45, 40, 44, 39, 43, 48, 3]]
# Max straight path length inside maze = 6 hexes. (illegal paths are 7+)
MAX_STRAIGHT_PATH_INSIDE_MAZE = 6
STRAIGHT_PATHS_INSIDE_MAZE = [[5, 7, 10, 13, 17, 21, 26, 31, 37, 42, 47],
                              [9, 12, 16, 20, 25, 30, 36, 41, 46],
                              [6, 8, 10, 13, 16, 20, 24, 29, 34, 39, 43],
                              [11, 14, 17, 21, 25, 30, 35, 40, 44],
                              [38, 32, 37, 31, 36, 30, 35, 29, 34, 28, 33],
                              [27, 22, 26, 21, 25, 20, 24, 19, 23]]

# Get all illegal straight paths to ports
illegal_straight_paths_list = []
for path in STRAIGHT_PATHS_TO_PORTS:
    for sub_path in get_subpaths(path, MAX_STRAIGHT_PATH_TO_PORT+1):
        illegal_straight_paths_list.append(sub_path)

# Store illegal straight paths as a set of tuples for O(1) lookup time
ILLEGAL_STRAIGHT_PATHS_TO_PORT = {tuple(path) for path in illegal_straight_paths_list}

# Get all illegal straight paths inside the maze
illegal_straight_paths_list = []
for path in STRAIGHT_PATHS_INSIDE_MAZE:
    for sub_path in get_subpaths(path, MAX_STRAIGHT_PATH_INSIDE_MAZE+1):
        illegal_straight_paths_list.append(sub_path)

# Store illegal straight paths as a set of tuples for O(1) lookup time
ILLEGAL_STRAIGHT_PATHS_INSIDE_MAZE = {tuple(path) for path in illegal_straight_paths_list}

################################# Define a bunch of functions #################################


############## Functions for generating a hex maze configuration ############## 

def add_edges_to_node(graph, node, edges):
    '''
    Add all edges to the specified node in the graph. 
    If the node does not yet exist in the graph, add the node.
    
    Args:
    graph (nx.Graph): The networkx graph object
    node: The node to add to the graph (if it does not yet exist)
    edges: The edges to the node in the graph
    '''
    for edge in edges:
        graph.add_edge(node, edge)


def create_empty_hex_maze():
    '''
    Use networkx to create a graph object representing the empty hex maze 
    before any barriers are added.
    
    Returns: 
    nx.Graph: A new networkx graph object representing all of 
    the hexes and potential transitions between them in the hex maze
    ''' 
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


def create_maze_graph(barrier_set):
    '''
    Given a set of barriers defining a hex maze configuration, 
    return a networkx graph object representing the maze.

    Args:
    barrier_set (set of ints): Set of hex locations
    where barriers are placed in this hex maze configuration
    
    Returns:
    nx.Graph: A networkx graph object representing the maze
    '''
    
    # Create a new empty hex maze object
    maze_graph = create_empty_hex_maze()
    
    # Remove the barriers
    for barrier in barrier_set:
        maze_graph.remove_node(barrier)
    return maze_graph


def get_critical_choice_points(maze):
    '''
    Given a barrier set or networkx graph representing the hex maze, 
    find all critical choice points between reward ports 1, 2, and 3.
    
    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR networkx graph object representing the maze
    
    Returns:
    set of ints: The critical choice points for this maze
    '''
    # If our input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze

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


def get_optimal_paths_between_ports(maze):
    '''
    Given a barrier set or networkx graph representing the hex maze,
    return a list of all optimal paths between reward ports in the maze.
    
    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR networkx graph object representing the maze

    Returns: 
    list of lists: A list of all optimal paths (in hexes) from 
    reward port 1 to 2, 1 to 3, and 2 to 3
    '''
    # If our input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze
    
    optimal_paths = []
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=1, target=2)))
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=1, target=3)))
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=2, target=3)))
    return optimal_paths


def get_optimal_paths(maze, start_hex, target_hex):
    '''
    Given a barrier set or networkx graph representing the hex maze,
    return a list of all optimal paths from the start_hex to the target_hex
    in the maze.
    
    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR networkx graph object representing the maze
    start_hex (int): The starting hex in the maze
    target_hex (int): The target hex in the maze

    Returns: 
    list of lists: A list of all optimal path(s) (in hexes) from 
    the start hex to the target hex
    '''
    # If our input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze
    
    return list(nx.all_shortest_paths(graph, source=start_hex, target=target_hex))


def get_reward_path_lengths(maze):
    '''
    Given a barrier set or networkx graph representing the hex maze, 
    get the minimum path lengths (in hexes) between reward ports 1, 2, and 3.

    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR networkx graph object representing the maze
    
    Returns:
    list: Reward path lengths in form [length12, length13, length23]
    '''

    # If our input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze

    # Get length of optimal paths between reward ports
    len12 = nx.shortest_path_length(graph, source=1, target=2)+1
    len13 = nx.shortest_path_length(graph, source=1, target=3)+1
    len23 = nx.shortest_path_length(graph, source=2, target=3)+1

    return [len12, len13, len23]


def get_path_independent_hexes_to_port(maze, reward_port):
    '''
    Find all path-independent hexes to a reward port, defined as hexes 
    that a rat MUST run through to get to the port regardless of which 
    path he is taking/his reward port of origin. These are the same as
    the hexes the rat must run through when leaving this port before he
    reaches the (first) critical choice point. 
    
    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR networkx graph object representing the maze
    reward_port (int): The reward port: 1, 2, or 3
    
    Returns:
    set of ints: The path-independent hexes the rat must always run
    through when going to and from this reward port
    '''
    # If our input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze

    # Get all shortest paths between reward_port and the other 2 ports
    other_ports = [1, 2, 3]
    other_ports.remove(reward_port)
    paths_a = list(nx.all_shortest_paths(graph, source=reward_port, target=other_ports[0]))
    paths_b = list(nx.all_shortest_paths(graph, source=reward_port, target=other_ports[1]))

    # The path-independent hexes are the common hexes on the shortest 
    # paths between the reward port and both other ports
    path_independent_hexes = set()
    for path_a in paths_a:
        for path_b in paths_b:
            shared_path = [hex for hex in path_a if hex in path_b]
            path_independent_hexes.update(shared_path)

    return path_independent_hexes


def get_hexes_from_port(maze, start_hex, reward_port):
    '''
    Find the minimum number of hexes from a given hex to a
    chosen reward port for a given maze configuration.
    
    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR a networkx graph object representing the maze
    start_hex (int): The hex to calculate distance from
    reward_port (int): The reward port: 1, 2, or 3
    
    Returns:
    int: The number of hexes from start_hex to reward_port
    '''
    # If our maze input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze

    # Get the shortest path length between start_hex and the reward port
    return nx.shortest_path_length(graph, source=start_hex, target=reward_port)


def has_illegal_straight_path(maze):
    '''
    Given a barrier set or networkx graph representing the hex maze,
    checks if there are any illegal straight paths.
    
    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR networkx graph object representing the maze

    Returns: 
    The (first) offending path, or False if none
    '''
    # If our input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze
    
    # Get optimal paths between reward ports
    optimal_paths = get_optimal_paths_between_ports(graph)

    # We do 2 separate checks here beacause we may have different path length critera
    # for paths to reward ports vs inside the maze
    
    # First check all subpaths against illegal paths to a reward port
    subpaths1 = set()
    for path in optimal_paths:
        subpaths1.update(get_subpaths(path, MAX_STRAIGHT_PATH_TO_PORT+1))

    for path in subpaths1:
        if path in ILLEGAL_STRAIGHT_PATHS_TO_PORT:
            return path # (equivalent to returning True)
        
    # Now check all subpaths against illegal paths inside the maze
    subpaths2 = set()
    for path in optimal_paths:
        subpaths2.update(get_subpaths(path, MAX_STRAIGHT_PATH_INSIDE_MAZE+1))

    for path in subpaths2:
        if path in ILLEGAL_STRAIGHT_PATHS_INSIDE_MAZE:
            return path # (equivalent to returning True)
    
    # If we did all of those checks and found no straight paths, we're good to go!
    return False


def is_valid_maze(maze, complain=False):
    '''
    Given a a barrier set or networkx graph representing a possible hex maze
    configuration, check if it is valid using the following criteria: 
    - there are no unreachable hexes (this also ensures all reward ports are reachable)
    - path lengths between reward ports are between 15-25 hexes
    - all critical choice points are >=6 hexes away from a reward port
    - there are a maximum of 3 critical choice points
    - no straight paths >MAX_STRAIGHT_PATH_TO_PORT hexes to reward port (including port hex)
    - no straight paths >STRAIGHT_PATHS_INSIDE_MAZE in middle of maze
    
    Args:
    maze (set OR nx.Graph): Set of barriers representing the hex maze
    OR networkx graph object representing the maze
    complain (bool): Optional. If our maze configuration is invalid, 
    print out the reason why. Defaults to False
    
    Returns: 
    True if the hex maze is valid, False otherwise
    '''
    # If our input is a barrier set, get the graph representation
    if isinstance(maze, (set, frozenset, list)):
        graph = create_maze_graph(maze)
    # If our input is already a graph, use that
    elif isinstance(maze, nx.Graph):
        graph = maze

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


def generate_good_maze():
    '''
    Generates a "good" hex maze as defined by the function is_valid_maze.
    Uses a naive generation approach (randomly generates sets of 9 barriers
    until we get a valid maze configuration).

    Returns: 
    set: the set of barriers defining the hex maze
    '''
    # Create the empty hex maze
    start_maze = create_empty_hex_maze()
    barriers = set()

    # Generate a set of 9 random barriers until we get a good maze
    is_good_maze = False
    while not is_good_maze:
        # Start with an empty hex maze (no barriers)
        test_maze = start_maze.copy()

        # Randomly select 9 barriers
        barriers = set(np.random.choice(POSSIBLE_BARRIERS, size=9, replace=False))

        # Add the barriers to the empty maze
        for barrier in barriers:
            test_maze.remove_node(barrier)

        # Check if this is a good maze
        is_good_maze = is_valid_maze(test_maze)

    return barriers


############## Functions for generating a next good barrier set given an initial barrier set ############## 

def single_barrier_moved(maze_1, maze_2):
    ''' Check if two mazes differ by the movement of a single barrier. 
    
    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze

    Returns:
    True if the mazes differ by the movement of a single barrier, False otherwise
    '''
    
    # The symmetric difference (XOR) between the sets must have exactly two elements
    # because each set should have exactly one barrier not present in the other set
    return len(maze_1.symmetric_difference(maze_2)) == 2


def have_common_path(paths_1, paths_2):
    '''
    Given 2 lists of hex paths, check if there is a common path between the 2 lists.
    Used for determining if there are shared optimal paths between mazes.
    
    Args:
    paths_1 (list of lists): List of optimal hex paths between 2 reward ports
    paths_2 (list of lists): List of optimal hex paths between 2 reward ports

    Returns:
    True if there is a common path between the 2 lists of paths, False otherwise.
    '''
    
    # Convert the path lists to tuples to make them hashable and store them in sets
    pathset_1 = set(tuple(path) for path in paths_1)
    pathset_2 = set(tuple(path) for path in paths_2)
    
    # Return True if there is 1 or more common path between the path sets, False otherwise
    return len(pathset_1.intersection(pathset_2)) > 0


def have_common_optimal_paths(maze_1, maze_2):
    '''
    Given the hex maze database and 2 mazes, check if the 2 mazes have at
    least one common optimal path between every pair of reward ports (e.g. the mazes
    share an optimal path between ports 1 and 2, AND ports 1 and 3, AND ports 2 and 3), 
    meaning the rat could be running the same paths even though the mazes are "different".
    
    (The result of this function is equivalent to checking if num_hexes_different_on_optimal_paths == 0)

    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns:
    True if the mazes have a common optimal path between all pairs of reward ports, False otherwise
    '''
    # Do these barrier sets have a common optimal path from port 1 to port 2?
    have_common_path_12 = have_common_path(
        get_optimal_paths(maze_1, start_hex=1, target_hex=2), 
        get_optimal_paths(maze_2, start_hex=1, target_hex=2))
    # Do these barrier sets have a common optimal path from port 1 to port 3?
    have_common_path_13 = have_common_path(
        get_optimal_paths(maze_1, start_hex=1, target_hex=3), 
        get_optimal_paths(maze_2, start_hex=1, target_hex=3))
    # Do these barrier sets have a common optimal path from port 2 to port 3?
    have_common_path_23 = have_common_path(
        get_optimal_paths(maze_1, start_hex=2, target_hex=3), 
        get_optimal_paths(maze_2, start_hex=2, target_hex=3))
    
    # Return True if the barrier sets have a common optimal path between all pairs of reward ports
    return (have_common_path_12 and have_common_path_13 and have_common_path_23)


def min_hex_diff_between_paths(paths_1, paths_2):
    '''
    Given 2 lists of hex paths, return the minimum number of hexes that differ 
    between the most similar paths in the 2 lists.
    Used for determining how different optimal paths are between mazes.
    
    Args:
    paths_1 (list of lists): List of optimal hex paths between 2 reward ports
    paths_2 (list of lists): List of optimal hex paths between 2 reward ports
    
    Returns:
    num_different_hexes (int): the min number of hexes different between a
    hex path in paths1 and a hex path in paths2 (hexes on path1 not on path2 
    + hexes on path2 not on path1). If there is 1 or more shared
    path between the path lists, the hex difference is 0.
    '''
    
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


def hexes_different_on_optimal_paths(maze_1, maze_2):
    '''
    Given 2 mazes, find the set of hexes different on optimal 
    paths between every pair of reward ports. This helps us quantify
    how different two maze configurations are.
    
    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns:
    hexes_on_optimal_paths_maze_1_not_2 (set): The set of hexes on optimal paths in maze_1 but 1 not maze_2
    hexes_on_optimal_paths_maze_2_not_1 (set): The set of hexes on optimal paths in maze_2 but 1 not maze_1
    '''

    # Get the set of all hexes on optimal paths for maze 1
    hexes_on_optimal_paths_maze1 = {hex for path in get_optimal_paths_between_ports(maze_1) for hex in path}
    
    # Get the set of all hexes on optimal paths for maze 2
    hexes_on_optimal_paths_maze2 = {hex for path in get_optimal_paths_between_ports(maze_2) for hex in path}
    
    # Return hexes exlusively on optimal paths in maze 1, and hexes exclusively on optimal paths in maze 2
    hexes_on_optimal_paths_maze_1_not_2 = hexes_on_optimal_paths_maze1 - hexes_on_optimal_paths_maze2
    hexes_on_optimal_paths_maze_2_not_1 = hexes_on_optimal_paths_maze2 - hexes_on_optimal_paths_maze1
    return hexes_on_optimal_paths_maze_1_not_2, hexes_on_optimal_paths_maze_2_not_1


def num_hexes_different_on_optimal_paths(maze_1, maze_2):
    '''
    Given 2 mazes, find the numer of hexes different on the optimal 
    paths between every pair of reward ports. This difference is equal to
    (# of hexes on optimal paths in maze_1 but not maze_2 + 
    # # of hexes on optimal paths in maze_2 but not maze_1)
    
    This helps us quantify
    how different two maze configurations are.
    
    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns:
    num_different_hexes (int): The number of hexes different on optimal paths
    between reward ports for maze_1 and maze_2
    '''

    # Get the hexes different on optimal paths between these 2 mazes
    hexes_maze1_not_maze2, hexes_maze2_not_maze1 = hexes_different_on_optimal_paths(maze_1, maze_2)

    # Return the combined number of hexes different on the optimal paths
    return len(hexes_maze1_not_maze2 | hexes_maze2_not_maze1)


def num_hexes_different_on_optimal_paths_OLD(maze_1, maze_2):
    '''
    *** DEPRECATED: We have moved away from this method of counting in favor of 
    counting all hexes different on optimal paths between ports exactly once 
    (regardless of if they appear on multiple optimal paths). 

    Keeping this function for now for posterity. Note that for barrier sequences generated prior
    to this deprecation, the `min_hex_diff` count used in `get_barrier_sequence` and related
    functions refers to the count using this method. ***

    Given 2 mazes, find the number of hexes different between optimal paths 
    between every pair of reward ports. Note that this version of the function
    "double-counts" path-independent hexes. For example, hexes that are different 
    on a path to port 1 are double counted because they are different on the 
    optimal path from port 1 to port 2 AND different on the optimal path from port 1 to port 3. 
    This helps us quantify how different two maze configurations are.
    
    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns:
    num_different_hexes (int): The min number of hexes different in the most 
    similar optimal paths between all reward ports for the 2 mazes
    '''
    # How many hexes different are the most similar optimal paths from port 1 to port 2?
    num_hexes_different_12 = min_hex_diff_between_paths(
        get_optimal_paths(maze_1, start_hex=1, target_hex=2), 
        get_optimal_paths(maze_2, start_hex=1, target_hex=2))
    # How many hexes different are the most similar optimal paths from port 1 to port 3?
    num_hexes_different_13 = min_hex_diff_between_paths(
        get_optimal_paths(maze_1, start_hex=1, target_hex=3), 
        get_optimal_paths(maze_2, start_hex=1, target_hex=3))
    # How many hexes different are the most similar optimal paths from port 2 to port 3?
    num_hexes_different_23 = min_hex_diff_between_paths(
        get_optimal_paths(maze_1, start_hex=2, target_hex=3), 
        get_optimal_paths(maze_2, start_hex=2, target_hex=3))
    
    # Return the total number of hexes different between the most similar optimal
    # paths between all 3 reward ports
    return (num_hexes_different_12 + num_hexes_different_13 + num_hexes_different_23)


def at_least_one_path_shorter_and_longer(maze_1, maze_2):
    ''' 
    Given 2 mazes, check if at least one optimal path between reward ports
    is shorter AND at least one is longer in one of the mazes compared to the other
    (e.g. the path length between ports 1 and 2 increases and the path length 
    between ports 2 and 3 decreases.

    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns: 
    True if at least one path is shorter AND at least one is longer, False otherwise
    '''
    # Get path lengths between reward ports for each barrier set
    paths_1 = get_reward_path_lengths(maze_1)
    paths_2 = get_reward_path_lengths(maze_2)
    
    # Check if >=1 path is longer and >=1 path is shorter
    return (any(a < b for a, b in zip(paths_1, paths_2)) and any(a > b for a, b in zip(paths_1, paths_2)))


def optimal_path_order_changed(maze_1, maze_2):
    ''' 
    Given 2 mazes, check if the length order of the optimal paths
    between reward ports has changed (e.g. the shortest path between reward ports
    used to be between ports 1 and 2 and is now between ports 2 and 3, etc.)

    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns: 
    True if the optimal path length order has changed, False otherwise
    '''
    
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


def no_common_choice_points(maze_1, maze_2):
    ''' 
    Given 2 mazes, check that there are no common choice points between them.

    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns: 
    True if there are no common choice points, False otherwise
    '''
    
    # Get the choice points for each barrier set
    choice_points_1 = get_critical_choice_points(maze_1)
    choice_points_2 = get_critical_choice_points(maze_2)
    
    # Check if there are no choice points in common
    return choice_points_1.isdisjoint(choice_points_2)


def get_barrier_change(maze_1, maze_2):
    '''
    Given 2 mazes that differ by the movement of a single barrier, 
    find the barrier that was moved.
    
    Args:
    maze_1 (set/frozenset): Set of barriers representing the first hex maze
    maze_2 (set/frozenset): Set of barriers representing the second hex maze
    
    Returns:
    old_barrier (int): The hex location of the barrier to be moved in the first set
    new_barrier (int): The hex location the barrier was moved to in the second set
    '''
    # Find the original barrier location
    old_barrier = maze_1 - maze_2

    # Find the new barrier location
    new_barrier = maze_2 - maze_1
    
    # Return as integers instead of sets/frozensets with a single element
    return next(iter(old_barrier)), next(iter(new_barrier))


def get_barrier_changes(barrier_sequence):
    '''
    Given a sequence of barrier sets that each differ by the movement of 
    a single barrier, find the barriers moved from each barrier set to the next.
    
    Args:
    barrier_sequence (list of sets): List of sequential barrier sets
    
    Returns:
    list of lists: A list of [old barrier, new barrier] defining each 
    transition between barrier sets
    '''
    barrier_changes = []
    for i in range(len(barrier_sequence) - 1):
        old_barrier, new_barrier = get_barrier_change(barrier_sequence[i], barrier_sequence[i+1])
        barrier_changes.append([old_barrier, new_barrier])
    return barrier_changes


# def get_next_barrier_sets(df, original_barriers, criteria=['one_path_shorter_and_longer', 'optimal_path_order_changed', 'no_common_choice_points'], criteria_type='ANY'):
#     '''
#     Given the hex maze database (df) and set of original barriers, get a list 
#     of next barrier sets created by the movement of a single barrier. The next
#     barrier set must not have the same optimal paths between all reward ports. 
    
#     Option to specify additional criteria (as a list of strings):
#     - 'one_path_shorter_and_longer': at least one path increases in length 
#     and another decreases in length compared to the original barrier set.
#     - 'optimal_path_order_changed': the length order of the optimal paths
#     between reward ports has changed (e.g. the shortest path between reward ports
#     used to be between ports 1 and 2 and is now between ports 2 and 3, etc.)
#     - 'no_common_choice_points': the 2 configurations have no choice points
#     in common
    
#     Option to specify criteria type:
#     - 'ANY' (default): the next barrier set is valid if it satisfies ANY of the criteria
#     - 'ALL': the next barrier set is valid if it satisfies ALL of the criteria
    
#     Returns:
#     list of sets: a list of potential new barrier sets
#     '''
    
#     criteria_functions = {
#         "one_path_shorter_and_longer": partial(at_least_one_path_shorter_and_longer, df, original_barriers),
#         "optimal_path_order_changed": partial(optimal_path_order_changed, df, original_barriers),
#         "no_common_choice_points": partial(no_common_choice_points, df, original_barriers)
#     }
    
#     # Find other valid mazes in the df that differ by the movement of a single barrier
#     potential_new_barriers = [b for b in df['barriers'] if single_barrier_moved(b, original_barriers)]
    
#     # Set up a list for the ones that meet our criteria
#     new_barriers = []
    
#     # For each potential new barrier set, make sure it meets all of our criteria
#     for bar in potential_new_barriers:
#         # Ensure the next barrier set doesn't have the same optimal paths between all reward ports
#         if have_common_optimal_paths(df, original_barriers, bar):
#             continue
            
#         # Check our other criteria
#         new_maze_meets_criteria = False
#         if criteria_type == "ALL":
#             new_maze_meets_criteria = all(criteria_functions[criterion](bar) for criterion in criteria)
#         else: # if not specified as "ALL", I choose to assume ANY
#             new_maze_meets_criteria = any(criteria_functions[criterion](bar) for criterion in criteria)
        
#         # If our new maze met all of the criteria, add it!
#         if new_maze_meets_criteria:
#             new_barriers.append(bar)
            
#     return new_barriers


def get_next_barrier_sets(df, original_barriers, criteria_type='ALL'):
    '''
    Given the hex maze database (df) and set of original barriers, get a list 
    of next barrier sets created by the movement of a single barrier. 
    
    We have 2 criteria:
    1. At least one path must be longer and one must be shorter.
    2. The optimal path order must have changed (the pair of reward ports that 
    used to be the closest together or furthest apart is now different).
    
    Optional argument criteria_type:
    criteria_type='ANY': (default) Accept new barrier sets that meet EITHER of these criteria
    criteria_type='ALL': Accept new barrier sets that meet BOTH of these criteria
    criteria_type='JOSE': Meet both of the above criteria + optimal path lengths are 17, 19, 21
    + only 1 choice point
    
    Returns:
    list of sets: a list of potential new barrier sets
    '''
    
    # Find other valid mazes in the df that differ by the movement of a single barrier
    potential_new_barriers = [b for b in df['barriers'] if single_barrier_moved(b, original_barriers)]
    
    # Set up a list for the ones that meet our criteria
    new_barriers = []
    
    # Check each potential new barrier set
    for bar in potential_new_barriers:      
        # Check if at least one path gets longer and at least one path gets shorter 
        criteria1 = at_least_one_path_shorter_and_longer(original_barriers, bar)
        # Check if the optimal path order has changed
        criteria2 = optimal_path_order_changed(original_barriers, bar)
        # Make sure the optimal path lengths are 17, 19, 21 (in any order)
        criteria3 = (set(df_lookup(df, bar, 'reward_path_lengths')) == {17, 19, 21})
        # Only 1 critical choice point
        criteria4 = df_lookup(df, bar, 'num_choice_points')==1
        
        # Accept the potential new barrier set if it meets our criteria
        if criteria_type=='ALL':
            if (criteria1 and criteria2):
                bar = frozenset(int(b) for b in bar) # make int instead of np.int64
                new_barriers.append(bar)
        elif criteria_type=='JOSE':
            if (criteria1 and criteria2 and criteria3 and criteria4):
                bar = frozenset(int(b) for b in bar) # make int instead of np.int64
                new_barriers.append(bar)
        else: # I choose to assume 'ANY' as the default
            if (criteria1 or criteria2): 
                bar = frozenset(int(b) for b in bar) # make int instead of np.int64
                new_barriers.append(bar)
    
    return new_barriers


def get_best_next_barrier_set(df, original_barriers):
    '''
    Given the hex maze database and an original barrier set, find the best 
    potential next barrier set (based on the number of hexes
    different on the optimal paths between reward ports).
    
    Args:
    df (dataframe): The database of all possible maze configurations.
    original_barriers (set): The initial barrier set.

    Returns:
    set: The "best" potential next barrier set (maximally different from the 
    original barrier set)
    '''
    
    # Get all potential next barrier sets (that differ from the original by the movement of a single barrier)
    potential_next_barriers = get_next_barrier_sets(df, original_barriers, criteria_type='ALL')
    
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


def find_all_valid_barrier_sequences(df, start_barrier_set, min_hex_diff=8, max_sequence_length=5):
    '''
    Finds all valid sequences of barriers starting from the given start_barrier_set.

    This function recursively generates all sequences of barrier sets where each barrier set
    in the sequence differs from the previous by the movement of a single barrier.
    The optimal paths that the rat can travel between reward ports must be different
    for all barrier sets in a sequence.

    Args:
    df (dataframe): The database of all possible maze configurations.
    start_barrier_set (set): The initial barrier set to start generating sequences from.
    min_hex_diff (int): The minimum combined number of hexes different between the most 
    similar optimal paths between all 3 reward ports for all mazes in a sequence.
    max_sequence_length (int): The maximum length of a sequence to generate.

    Returns:
    list of list of sets: A list of all valid sequences of barriers. Each sequence
    is represented as a list of barrier sets.
    '''
    
    def helper(current_barrier_set, visited, current_length):
        '''
        A helper function to recursively find all valid sequences of barrier sets.
        The "visited" set ensures that no barrier set is revisited to avoid cycles/repetitions.

        Args:
        current_barrier_set (set): The current barrier set being processed.
        visited (set): A set of barrier sets that have already been visited to avoid cycles.
        current_length (int): The current length of our generated sequence.

        Returns:
        list of list of sets: A list of all valid barrier sequences starting from the 
        current_barrier_set. Each sequence is represented as a list of barrier sets.
        '''
        #print(f"Current set: {current_barrier_set}")
        #print(f"Visited: {visited}")
        
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
        next_sets = [s for s in next_sets if all(num_hexes_different_on_optimal_paths(s, v)>=min_hex_diff for v in visited)]
        
        # Initialize a list to store sequences
        sequences = []
        
        # Iterate over each next valid set
        for next_set in next_sets:
            if next_set not in visited:
                # Mark the next set as visited
                visited.add(next_set)
                
                # Recursively find sequences from the next set
                subsequences = helper(next_set, visited, current_length+1)
                
                # Append the current set to the beginning of each subsequence
                for subsequence in subsequences:
                    sequences.append([current_barrier_set] + subsequence)
                
                # Unmark the next set as visited (backtrack)
                visited.remove(next_set)
        
        # If no valid sequences were found, return the current barrier set as the only sequence
        if not sequences:
            return [[current_barrier_set]]
        
        return sequences
    
    # Start the recursive search from the initial barrier set
    return helper(start_barrier_set, {frozenset(start_barrier_set)}, 1)


def get_barrier_sequence(df, start_barrier_set, min_hex_diff=8, max_sequence_length=5, max_recursive_calls=40, criteria_type='ANY'):
    '''
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

    Args:
    df (dataframe): The database of all possible maze configurations.
    start_barrier_set (set): The initial barrier set to start generating a sequence from.
    min_hex_diff (int): The minimum combined number of hexes different between the most 
    similar optimal paths between all 3 reward ports for all mazes in a sequence (default=8).
    max_sequence_length (int): The maximum length of a sequence to generate. Will stop 
    searching and automatically return a sequence once we find one of this length (default=5).
    max_recursive_calls (int): The maximum number of recursive calls to make on our search
    for a good barrier sequence (so the function doesn't run for a really long time). Stops 
    the search and returns the longest valid barrier sequence found by this point (default=40).
    criteria_type (String): The criteria type for what makes a valid next barrier set to propagate
    to get_next_barrier_sets. Options are 'ALL', 'ANY', or 'JOSE'. Defaults to 'ANY'

    Returns:
    list of sets: A valid sequence of barrier sets that is "good enough" (meaning it
    fulfills all our criteria but is not necessarily the best one), or the starting barrier
    set if no such sequence is found.
    '''
    
    # Keep track of our longest sequence found in case we don't find one of max_sequence_length
    longest_sequence_found = []
    
    # Stop looking and return the longest sequence found after max_recursive_calls (for speed)
    recursive_calls = 0
    
    def helper(current_sequence, visited, current_length):
        '''
        A helper function to recursively find a "good enough" sequence of barrier sets.
        The "visited" set ensures that no barrier set is revisited to avoid cycles/repetitions.

        Args:
        current_sequence (list of sets): The current sequence of barrier sets being processed.
        visited (set): A set of barrier sets that have already been visited to avoid cycles.
        current_length (int): The current length of the sequence.

        Returns:
        list of sets: A valid sequence of barrier sets that is "good enough" (meaning it
        fulfills all our criteria but is not necessarily the best one), or the current
        barrier sequence if no such sequence is found.
        '''
        # We keep track of these outside of the helper function
        nonlocal longest_sequence_found, recursive_calls
        
        # Keep track of how many times we have called the helper function
        recursive_calls += 1
        
        #print("in helper")
        # Base case: if the sequence length has reached the maximum, return the current sequence
        if current_length >= max_sequence_length:
            return current_sequence
       
        #print(f"Current sequence: {current_sequence}")
        
        # If this sequence is longer than our longest sequence found, it is our new longest sequence
        if current_length > len(longest_sequence_found):
            #print("This is our new longest sequence!")
            longest_sequence_found = current_sequence
            
        # If we have reached the limit of how many times to call helper, return the longest sequence found
        if recursive_calls >= max_recursive_calls:
            #print("Max recursive calls reached!")
            return longest_sequence_found
        
        current_barrier_set = current_sequence[-1]
        
        # Search the database for all valid new barrier sets from the current barrier set
        next_sets = get_next_barrier_sets(df, current_barrier_set, criteria_type=criteria_type)
        
        # Remove the current barrier set from the next sets to avoid self-referencing
        next_sets = [s for s in next_sets if s != current_barrier_set]
        
        # Remove barrier sets with optimal paths too similar to any other barrier set in the sequence
        next_sets = [s for s in next_sets if all(num_hexes_different_on_optimal_paths(s, v)>=min_hex_diff for v in visited)]
        
        # Iterate over each next valid set
        for next_set in next_sets:
            if next_set not in visited:
                # Mark the next set as visited
                visited.add(next_set)
                
                # Recursively find sequences from the next set
                result = helper(current_sequence + [next_set], visited, current_length+1)
                
                # If a sequence of the maximum length is found, return it
                if result and len(result) == max_sequence_length:
                    return result
                
                # Unmark the next set as visited (backtrack)
                visited.remove(next_set)
        
        # If no valid sequences were found, return the current sequence
        #print(f"Sequence at return: {current_sequence}")
        return current_sequence
    
    # Start the recursive search from the initial barrier set
    barrier_sequence = helper([start_barrier_set], {frozenset(start_barrier_set)}, 1)
    
    #print(f"Barrier sequence: {barrier_sequence}")
    #print(f"Longest sequence: {longest_sequence_found}")
    
    # Return the longest sequence
    return longest_sequence_found if len(longest_sequence_found) > len(barrier_sequence) else barrier_sequence


############## Functions for maze rotations and relfections across its axis of symmetry ############## 

def rotate_hex(original_hex, direction='counterclockwise'):
    '''
    Given a hex in the hex maze, returns the corresponding hex if the maze is rotated once
    counterclockwise (e.g. hex 1 becomes hex 2, 4 becomes 49, etc.). Option to specify
    direction='clockwise' to rotate clockwise instead (e.g 1 becomes 3, 4 becomes 48, etc.)

    Args:
    original_hex (int): The hex in the hex maze to rotate (1-49)
    direction (String): Which direction to rotate the hex ('clockwise' or 'counterclockwise')
    Defaults to 'counterclockwise'
    
    Returns: 
    int: The corresponding hex if the maze was rotated once in the specified direction
    '''
    # Lists of corresponding hexes when the maze is rotated 120 degrees
    hex_rotation_lists = [[1,2,3], [4,49,48], [6,47,33], [5,38,43], [8,42,28], 
                         [7,32,39], [11,46,23], [10,37,34], [9,27,44], [14,41,19],
                         [13,31,29], [12,22,40], [18,45,15], [17,36,24], [16,26,35],
                         [21,30,20], [25]]
    
    for lst in hex_rotation_lists:
        if original_hex in lst:
            index = lst.index(original_hex)
            if direction=='clockwise':
                return lst[(index - 1) % len(lst)]
            else: # I choose to assume any direction not specified 'clockwise' is 'counterclockwise'
                return lst[(index + 1) % len(lst)]
    # Return None if the hex to rotate doesn't exist in our rotation lists (all hexes should exist)
    return None  


def reflect_hex(original_hex, axis=1):
    '''
    Given a hex in the hex maze, returns the corresponding hex if the maze is reflected
    across the axis of hex 1 (e.g. hex 6 becomes hex 5 and vice versa, 8 becomes 7, etc.). 
    Option to specify axis=2 or axis=3 to reflect across the axis of hex 2 or 3 instead.

    Args:
    original_hex (int): The hex in the maze to reflect (1-49)
    axis (int): Which reward port axis to reflect the maze across. Must be
    1, 2, or 3. Defaults to 1
    
    Returns: 
    int: The corresponding hex if the maze was reflected across the specified axis
    '''
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
    

def get_rotated_barriers(original_barriers, direction='counterclockwise'):
    '''
    Given a set of barriers in the hex maze, returns the corresponding 
    barrier set if the maze is rotated once counterclockwise (e.g. hex 1 becomes hex 2, 
    4 becomes 49, etc.). Option to specify direction='clockwise' to rotate clockwise 
    instead (e.g 1 becomes 3, 4 becomes 48, etc.)

    Args:
    original_barriers (set/frozenset): A set of barriers defining a hex maze
    direction (String): Which direction to rotate the maze ('clockwise' or 'counterclockwise')
    Defaults to 'counterclockwise'
    
    Returns: 
    set: The barrier set if the maze was rotated once in the specified direction
    '''
    return {rotate_hex(b, direction) for b in original_barriers}


def get_reflected_barriers(original_barriers, axis=1):
    '''
    Given a set of barriers in the hex maze, returns the corresponding 
    barrier set if the maze is reflected along the axis of hex 1 
    (e.g. hex 6 becomes hex 5 and vice versa, 8 becomes 7 and vice versa, etc.). 
    Option to specify axis=2 or axis=3 to reflect across the axis of hex 2 or 3 instead.

    Args:
    original_barriers (set/frozenset): A set of barriers defining a hex maze
    axis (int): Which reward port axis to reflect the maze across. Must be
    1, 2, or 3. Defaults to 1
    
    Returns: 
    set: The barrier set if the maze was reflected across the specified axis
    '''
    return {reflect_hex(b, axis) for b in original_barriers}


def get_isomorphic_mazes(maze):
    '''
    Given a set of barriers defining a hex maze configuration, return the
    other 5 mazes that have the same graph structure (corresponding
    to the maze rotated clockwise/counterclockwise and reflected across its
    3 axes of symmetry)

    Args:
    maze (set/frozenset): A set of barriers defining a hex maze
    
    Returns:
    set of frozensets: a set of the 5 barrier sets defining mazes isomorphic 
    to this maze
    '''
    # Rotate and reflect the maze to get other barrier configs that 
    # represent the same underlying graph structure
    reflected_ax1 = frozenset(get_reflected_barriers(maze, axis=1))
    reflected_ax2 = frozenset(get_reflected_barriers(maze, axis=2))
    reflected_ax3 = frozenset(get_reflected_barriers(maze, axis=3))
    rotated_ccw = frozenset(get_rotated_barriers(maze, direction='counterclockwise'))
    rotated_cw = frozenset(get_rotated_barriers(maze, direction='clockwise'))
    
    return {reflected_ax1, reflected_ax2, reflected_ax3, rotated_ccw, rotated_cw}


############## Use the above functions to get all the info about a maze configuration ##############

def df_lookup(df, barriers, attribute_name):
    ''' 
    Use the dataframe to look up a specified attribute of a barrier set. 
    
    Args:
    df (DataFrame): The hex maze database
    barriers (set/frozenset): A set of barriers defining a hex maze
    attribute_name (String): The maze attribute to look up in the df.
    Must exist as a column in the df

    Returns:
    The value of the attribute for this maze, or None if the maze isn't in the df
    '''
    # Check if the attribute_name exists as a column in the DataFrame
    if attribute_name not in df.columns:
        raise ValueError(f"Column '{attribute_name}' does not exist in the DataFrame.")
    
    # Filter the DataFrame
    filtered_df = df[df['barriers'] == barriers][attribute_name]

    # If this maze isn't found in the dataframe, return None
    if filtered_df.empty:
        return None
    # Otherwise return the value of the attribute
    else:
        return filtered_df.iloc[0]


def get_maze_attributes(barrier_set):
    '''
    Given a set of barriers defining a maze, create a dictionary of attributes for that maze.
    Includes the length of the optimal paths between reward ports, the optimal paths
    between these ports, ther path length difference between optimal paths, 
    critical choice points, the number of cycles and the hexes defining these cycles, 
    and a set of other maze configurations isomorphic to this maze.

    Args:
    barrier_set (set/frozenset): A set of barriers defining a hex maze
    
    Returns: 
    dict: A dictionary of attributes of this maze
    '''
    
    # Get the graph representation of the maze for us to do calculations on
    maze = create_maze_graph(barrier_set)

    # Get length of optimal paths between reward ports
    len12 = nx.shortest_path_length(maze, source=1, target=2)+1
    len13 = nx.shortest_path_length(maze, source=1, target=3)+1
    len23 = nx.shortest_path_length(maze, source=2, target=3)+1
    reward_path_lengths = [len12, len13, len23]
    path_length_difference = max(reward_path_lengths) - min(reward_path_lengths)
    
    # Get the optimal paths between reward ports
    optimal_paths_12 = list(nx.all_shortest_paths(maze, source=1, target=2))
    optimal_paths_13 = list(nx.all_shortest_paths(maze, source=1, target=3))
    optimal_paths_23 = list(nx.all_shortest_paths(maze, source=2, target=3))
    optimal_paths_all = []
    optimal_paths_all.extend(optimal_paths_12)
    optimal_paths_all.extend(optimal_paths_13)
    optimal_paths_all.extend(optimal_paths_23)
    
    # Get critical choice points
    choice_points = set(get_critical_choice_points(maze))
    num_choice_points = len(choice_points)
    
    # Get information about cycles
    cycle_basis = nx.cycle_basis(maze)
    num_cycles = len(cycle_basis)
    
    # Get a list of isomorphic mazes
    isomorphic_mazes = get_isomorphic_mazes(barrier_set)
    
    # Create a dictionary of attributes
    attributes = {'barriers': barrier_set, 'len12': len12, 'len13': len13, 'len23': len23, 
                  'reward_path_lengths': reward_path_lengths, 'path_length_difference': path_length_difference,
                  'optimal_paths_12': optimal_paths_12, 'optimal_paths_13': optimal_paths_13,
                  'optimal_paths_23': optimal_paths_23, 'optimal_paths_all': optimal_paths_all,
                  'choice_points': choice_points, 'num_choice_points': num_choice_points,
                 'cycles': cycle_basis, 'num_cycles': num_cycles, 'isomorphic_mazes':isomorphic_mazes}
    return attributes


def get_barrier_sequence_attributes(barrier_sequence):
    '''
    Given the maze configuration database (df) and a sequence of 
    maze configurations that differ by the movement of a single barrier, 
    get the barrier change between each maze, reward path lengths and 
    choice points for all mazes in the sequence,
    and return a dictionary of these attributes.
    
    Args:
    barrier_sequence (list of sets): The sequence of maze configurations.

    Returns:
    dict: A dictionary of attributes of this sequence.
    '''
    
    reward_path_lengths = []
    choice_points = []
    
    # Get attributes for each barrier set in the sequence
    for bars in barrier_sequence:
        reward_path_lengths.append(get_reward_path_lengths(bars))
        choice_points.append(get_critical_choice_points(bars))
    
    barrier_changes = get_barrier_changes(barrier_sequence)
    
    # Set up a dictionary of attributes
    barrier_dict = {'barrier_sequence': barrier_sequence, 
                    'sequence_length': len(barrier_sequence),
                    'barrier_changes': barrier_changes,
                    'reward_path_lengths': reward_path_lengths,
                    'choice_points': choice_points}
    return barrier_dict


################################ Plotting hex mazes ################################

def get_hex_centroids(scale=1):
    ''' 
    Calculate the (x,y) coordinates of each hex centroid.
    Centroids are calculated relative to the centroid of hex 1 at (0,0).

    Args:
    scale (int): The width of each hex (aka the length of the long diagonal, 
    aka 2x the length of a single side). Defaults to 1
    
    Returns:
    dict: a dictionary of hex: (x,y) coordinate of centroid
    '''

    # Number of hexes in each vertical row of the hex maze
    hexes_per_row = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    # List of hexes in order from top to bottom, left to right
    hex_list = [1, 4, 6, 5, 8, 7, 11, 10, 9, 14, 13, 12, 18, 17, 16, 15,
                22, 21, 20, 19, 27, 26, 25, 24, 23, 32, 31, 30, 29, 28,
                38, 37, 36, 35, 34, 33, 49, 42, 41, 40, 39, 48, 2, 47, 46, 45, 44, 43, 3]
    # Vertical distance between rows for touching hexes
    y_offset = math.sqrt(3) / 2  * scale
    y_shift = 0
    count = 0
    hex_positions = {}
    for row, hexes_in_this_row in enumerate(hexes_per_row):
        if row % 2 == 0 and row != 0:  # Every other row, shift an extra 1/2 hex down
            y_shift += y_offset / 2
        for hex in range(hexes_in_this_row):
            x = hex * 3/2 * scale - (hexes_in_this_row - 1) * 3/4 * scale
            y = -row * y_offset + y_shift
            hex_positions[hex_list[count]] = (x, y)
            count += 1
    return hex_positions


def get_stats_coords(hex_centroids):
    '''
    When plotting a hex maze with additional stats (such as path lengths), get the
    graph coordinates of where to display those stats based on the hex centroids.

    Args:
    hex_centroids (dict): Dictionary of hex_id: (x, y) centroid of that hex

    Returns:
    stats_coords (dict): Dictionary of stat_id: (x, y) coordinates of where to plot it.
    The stat_id should be the same as a key returned by `get_maze_attributes`
    '''
    stats_coords = {}

    # Height of a hex can be calculated by abs(y coord of hex 1 - y coord of hex 4)
    hex_height = abs(hex_centroids[1][1] - hex_centroids[4][1])

    # Path length for port 1 to port 2 is (x coord of hex 38, y coord of hex 18)
    stats_coords['len12'] = (hex_centroids[38][0], hex_centroids[18][1])

    # Path length for port 1 to port 3 is (x coord of hex 33, y coord of hex 15)
    stats_coords['len13'] = (hex_centroids[33][0], hex_centroids[15][1])

    # Path length for port 2 to port 3 is (x coord of hex 45, y coord of hex 45 - 1.5 hex)
    stats_coords['len23'] = (hex_centroids[45][0], hex_centroids[45][1] - 1.5*hex_height)

    return stats_coords



def plot_hex_maze(barriers=None, old_barrier=None, new_barrier=None, 
                  show_barriers=True, show_choice_points=True,
                  show_optimal_paths=False, show_arrow=True,
                  show_barrier_change=True, show_stats=False, scale=1):
    ''' 
    Given a set of barriers specifying a hex maze, plot the maze
    in classic hex maze style.
    Open hexes are shown in light blue. By default, barriers are shown
    in black, and choice point(s) are shown in yellow.
    
    Option to specify old_barrier hex and new_barrier hex 
    to indicate a barrier change configuration:
    The now-open hex where the barrier used to be is shown in pale red.
    The new barrier is shown in dark red. An arrow indicating the movement
    of the barrier from the old hex to the new hex is shown in pink.
    
    Args:
    barriers (set of ints): A set defining the hexes where barriers are placed in the maze. \
        If no barriers or 'None' is specifed, plots an empty hex maze
    old_barrier (int): Optional. The hex where the barrier was in the previous maze
    new_barrier (int): Optional. The hex where the new barrier is in this maze

    Additional args to change the plot style:
    - show_barriers (bool): If the barriers should be shown as black hexes and labeled. \
    If False, only open hexes are shown. Defaults to True
    - show_choice_points (bool): If the choice points should be shown in yellow. \
    If False, the choice points are not indicated on the plot. Defaults to True
    - show_optimal_paths (bool): Highlight the hexes on optimal paths between \
    reward ports in light green. Defaults to False
    - show_arrow (bool): Draw an arrow indicating barrier movement from the \
    old_barrier hex to the new_barrier hex. Defaults to True if old_barrier and \
    new_barrier are not None
    - show_barrier_change (bool): Highlight the old_barrier and new_barrier hexes \
    on the maze. Defaults to True if old_barrier and new_barrier are not None. \
    - show_stats (bool): Print maze stats (lengths of optimal paths between ports) \
    on the graph. Defaults to False
    Note that highlighting choice points takes precendence over barrier change \
    hexes, as they are also shown by the movement arrow. If show_barriers=False, \
    the new_barrier hex will not be shown (because no barriers are shown with this option.) 
    '''
    
    # Create an empty hex maze
    hex_maze = create_empty_hex_maze()
    # Get a dictionary of the (x,y) coordinates of each hex centroid
    hex_coordinates = get_hex_centroids(scale)
    # Get a dictionary of stats coordinates based on hex coordinates
    if show_stats:
        stats_coordinates = get_stats_coords(hex_coordinates)
    # Define this for times we want to draw the arrow but not show barriers
    new_barrier_coords = None

    # Make the open hexes light blue
    hex_colors = {node: 'skyblue' for node in hex_maze.nodes()}

    if barriers is not None:
        # Make the barriers black if we want to show them
        if show_barriers:
            for hex in barriers:
                hex_colors.update({hex: 'black'})
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
                hex_colors.update({hex: 'lightgreen'})

        # Optional - Make the old barrier location (now an open hex) light red to indicate barrier change
        if old_barrier is not None and show_barrier_change:
            hex_colors.update({old_barrier: 'peachpuff'})

        # Optional - Make the new barrier location dark red to indicate barrier change
        if new_barrier is not None and show_barrier_change:
            hex_colors.update({new_barrier: 'darkred'})

        # Optional - Make the choice point(s) yellow
        if show_choice_points:
            choice_points = get_critical_choice_points(barriers)
            for hex in choice_points:
                hex_colors.update({hex: 'gold'})

    fig, ax = plt.subplots()
    # Add each hex to the plot
    for hex, (x, y) in hex_coordinates.items():
        hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=scale/2,
                                         orientation=math.pi/6, facecolor=hex_colors[hex], edgecolor='white')
        ax.add_patch(hexagon)

    # If we have a barrier change, add an arrow between the old_barrier and new_barrier to show barrier movement
    if show_arrow and old_barrier is not None and new_barrier is not None:
            arrow_start = hex_coordinates[old_barrier]
            # If we removed new_barrier from our dict (because show_barriers=False), use the saved coordinates
            arrow_end = hex_coordinates.get(new_barrier, new_barrier_coords)
            ax.annotate("",
                xy=arrow_end, xycoords='data',
                xytext=arrow_start, textcoords='data',
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3, rad=0.2",
                                color="salmon", linewidth=2))

    # Add hex labels
    nx.draw_networkx_labels(hex_maze, hex_coordinates, labels={h: h for h in hex_maze.nodes()}, font_color='black')

    # Add barrier labels
    if barriers is not None and show_barriers:
        nx.draw_networkx_labels(hex_maze, hex_coordinates, labels={b: b for b in barriers}, font_color='white')

    # Add stats
    if show_stats:
        # Get stats for this maaze
        maze_attributes = get_maze_attributes(barriers)
        # For all stats that we have display coordinates for, print them on the graph
        for stat in maze_attributes:
            if stat in stats_coordinates:
                ax.annotate(maze_attributes[stat], 
                            stats_coordinates[stat],
                            ha='center', # Horizontal alignment
                            fontsize=12  # Font size
                )

    # Adjust axes
    plt.xlim(-5.5*scale, 5.5*scale)
    plt.ylim(-9.5*scale, 1*scale) if show_stats else plt.ylim(-9*scale, 1*scale)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    

def plot_barrier_change_sequence(barrier_sequence, print_barrier_info=True,
                                 **kwargs):
    '''
    Given a sequence of barrier sets that each differ by the movement of 
    a single barrier, plot each maze in the sequence with the moved barriers
    indicated on each maze.
    
    Open hexes are shown in light blue. By default, barriers are shown
    in black, and choice point(s) are shown in yellow.
    The now-open hex where the barrier used to be is shown in pale red.
    The new barrier is shown in dark red. An arrow indicating the movement
    of the barrier from the old hex to the new hex is shown in pink.
    
    Args:
    barrier_sequence (list of sets): List of sequential barrier sets
    print_barrier_info (bool): Optional. Print each barrier set and the
    barrier moved between barrier sets. Defaults to True

    Additional args to change the plot style (same as for plot_hex_maze):
    - show_barriers (bool): If the barriers should be shown as black hexes and labeled. \
    If False, only open hexes are shown. Defaults to True
    - show_choice_points (bool): If the choice points should be shown in yellow. \
    If False, the choice points are not indicated on the plot. Defaults to True
    - show_optimal_paths (bool): Highlight the hexes on optimal paths between \
    reward ports in light green. Defaults to False
    - show_arrow (bool): Draw an arrow indicating barrier movement from the \
    old_barrier hex to the new_barrier hex. Defaults to True if old_barrier and \
    new_barrier are not None
    - show_barrier_change (bool): Highlight the old_barrier and new_barrier hexes \
    on the maze. Defaults to True if old_barrier and new_barrier are not None. \
    Note that highlighting choice points takes precendence over barrier change \
    hexes, as they are also shown by the movement arrow. If show_barriers=False, \
    the new_barrier hex will not be shown (because no barriers are shown with this option.) 
    '''
    
    # Find the barriers moved from one configuration to the next
    barrier_changes = get_barrier_changes(barrier_sequence)
    
    # First print and plot the first barrier set
    # Pass any additional style args directly to plot_hex_maze
    if (print_barrier_info):
        print(f"Barrier set 0: {barrier_sequence[0]}")
    plot_hex_maze(barrier_sequence[0], **kwargs)
    
    # Now print barrier change info and plot each successive barrier set
    for i, (barriers, (old_barrier_hex, new_barrier_hex)) in enumerate(zip(barrier_sequence[1:], barrier_changes)):
        if (print_barrier_info):
            print(f"Barrier change: {old_barrier_hex} -> {new_barrier_hex}")
            print(f"Barrier set {i+1}: {barriers}")
        plot_hex_maze(barriers, old_barrier=old_barrier_hex, new_barrier=new_barrier_hex, **kwargs)


def plot_hex_maze_networkx(barriers, old_barrier=None, new_barrier=None,
                           show_barriers=True, show_choice_points=True):
    ''' 
    ***** DEPRECATED: replaced by plot_hex_maze *****
    
    Given a set of barriers specifying a hex maze, plot the maze as a 
    networkx style graph with nodes connected by lines.
    Open hexes are shown in light blue, connected by thin grey lines.
    Barriers are shown in dark grey. Choice point(s) are in yellow.
    
    Option to specify old barrier location and new barrier location 
    to indicate a barrier change configuration:
    The now-open hex where the barrier used to be is shown in pale red.
    The new barrier is shown in dark red.
    
    Args:
    barriers (set): A set defining the hexes where barriers are placed in the maze
    old_barrier (int): Optional. The hex where the barrier was in the previous maze
    new_barrier (int): Optional. The hex where the new barrier is in this maze
    '''
    
    # Create an empty maze for graph layout
    base_hex_maze = create_empty_hex_maze()
    
    # Create our actual maze
    maze = base_hex_maze.copy()
    for hex in barriers:
        maze.remove_node(hex)

    # Get the graph layout of the original maze
    pos = nx.kamada_kawai_layout(base_hex_maze)

    # Draw the available hexes in our maze using this layout
    nx.draw(maze, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=400)

    # Add the barriers in black
    if show_barriers:
        nx.draw_networkx_nodes(base_hex_maze, pos, nodelist={b: b for b in barriers}, node_color='black', node_size=400)   
        nx.draw_networkx_labels(base_hex_maze, pos, labels={b: b for b in barriers}, font_color='white')

    # Make the choice point(s) yellow
    if show_choice_points:
        choice_points = get_critical_choice_points(maze)
        for choice_point in choice_points:
            nx.draw_networkx_nodes(base_hex_maze, pos, nodelist=[choice_point], node_color='yellow', node_size=400)
        
    # Make the old barrier location that is now an open hex light red
    if old_barrier is not None:
        nx.draw_networkx_nodes(base_hex_maze, pos, nodelist=[old_barrier], node_color='peachpuff', node_size=400)
    
    # Make the new barrier location dark red
    if new_barrier is not None:
        nx.draw_networkx_nodes(base_hex_maze, pos, nodelist=[new_barrier], node_color='darkred', node_size=400)
    
    plt.show()


############## One-time use functions to help ensure that our database includes all possible mazes ##############
    
def num_isomorphic_mazes_in_set(set_of_valid_mazes, maze):
    '''
    Given a set of all valid maze configurations and a set of barriers defining 
    a single hex maze configuration, find all isomorphic mazes for this 
    configuration that already exist in our larger set, and which are missing.

    Args:
    set_of_valid_mazes (set of frozensets): Set of all valid maze configurations
    maze (set/frozenset): Set of barriers defining a single maze configuration
    
    Returns:
    int: The number of isomorphic mazes that already exist in the set
    list of sets: A list of isomorphic maze configurations missing from the set
    '''
    # Get all potential isomorphic mazes for this barrier configuration
    all_isomorphic_barriers = get_isomorphic_mazes(maze)
    # Find other mazes in the dataframe that are isomorphic to the given barrier set
    isomorphic_barriers_in_set = set([b for b in set_of_valid_mazes if b in all_isomorphic_barriers])
    # Get the isomorphic mazes not present in the dataframe
    isomorphic_bariers_not_in_set = all_isomorphic_barriers.difference(isomorphic_barriers_in_set)
    return len(isomorphic_barriers_in_set), isomorphic_bariers_not_in_set


def num_isomorphic_mazes_in_df(df, maze):
    '''
    Given our maze configuration database and a set of barriers defining 
    a hex maze configuration, find all isomorphic mazes that already exist 
    in the dataframe, and which are missing.

    Args:
    df (DataFrame): DataFrame containing all valid maze configurations in the
    column 'barriers'
    maze (set/frozenset): Set of barriers defining a single maze configuration
    
    Returns:
    int: the number of isomorphic mazes that already exist in the dataframe
    list of sets: a list of isomorphic maze configurations missing from the dataframe
    '''
    # Get all potential isomorphic mazes for this barrier configuration
    all_isomorphic_barriers = get_isomorphic_mazes(maze)
    # Find other mazes in the dataframe that are isomorphic to the given barrier set
    isomorphic_barriers_in_df = set([b for b in df['barriers'] if b in all_isomorphic_barriers])
    # Get the isomorphic mazes not present in the dataframe
    isomorphic_bariers_not_in_df = all_isomorphic_barriers.difference(isomorphic_barriers_in_df)
    return len(isomorphic_barriers_in_df), isomorphic_bariers_not_in_df
