import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
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
    graph: the networkx graph object
    node: the node to add to the graph (if it does not yet exist)
    edges: the edges to the node in the graph
    '''
    for edge in edges:
        graph.add_edge(node, edge)


def create_empty_hex_maze():
    '''
    Use networkx to create a graph representing the empty hex maze before any barriers are added.
    
    Returns: 
    a new networkx graph object representing all of the hexes in the hex maze
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
    '''
    
    # Create a new empty hex maze object
    maze_graph = create_empty_hex_maze()
    
    # Remove the barriers
    for barrier in barrier_set:
        maze_graph.remove_node(barrier)
    return maze_graph


def find_all_critical_choice_points(graph):
    '''
    Given a networkx graph representing the hex maze, find all 
    critical choice points between reward ports 1, 2, and 3.
    
    Returns:
    set: the critical choice points for this maze
    '''
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


def has_illegal_straight_path(graph):
    '''
    Given a networkx graph of the hex maze, checks if there are any illegal straight paths.
    
    Returns: 
    the (first) offending path, or False if none
    '''
    optimal_paths = []
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=1, target=2)))
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=1, target=3)))
    optimal_paths.extend(list(nx.all_shortest_paths(graph, source=2, target=3)))

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


def is_valid_maze(graph, complain=False):
    '''
    Given a graph representing a possible hex maze configuration, check if it is valid 
    using the following criteria: 
    - there are no unreachable hexes (this also ensures all reward ports are reachable)
    - path lengths between reward ports are between 15-25 hexes
    - all critical choice points are >=6 hexes away from a reward port
    - there are a maximum of 3 critical choice points
    - no straight paths >MAX_STRAIGHT_PATH_TO_PORT hexes to reward port (including port hex)
    - no straight paths >STRAIGHT_PATHS_INSIDE_MAZE in middle of maze
    
    Optional argument complain (defaults to False):
    - When True: If our maze configuration is invalid, print out the reason why.
    
    Returns: 
    True if the hex maze is valid, False otherwise
    '''

    # Make sure all (non-barrier) hexes are reachable
    if not nx.is_connected(graph):
        if complain:
            print("BAD MAZE: At least one (non-barrier) hex is unreachable")
        return False
    
    # Make sure path lengths are between 15-25
    len12 = nx.shortest_path_length(graph, source=1, target=2)
    len13 = nx.shortest_path_length(graph, source=1, target=3)
    len23 = nx.shortest_path_length(graph, source=2, target=3)
    reward_port_lengths = [len12, len13, len23]
    if min(reward_port_lengths) <= 13:
        if complain:
            print("BAD MAZE: Path between reward ports is too short (<=13)")
        return False
    if max(reward_port_lengths) - min(reward_port_lengths) < 4:
        if complain:
            print("BAD MAZE: Distance difference in reward port paths is too small (<4)")
        return False
    if max(reward_port_lengths) > 25:
        if complain:
            print("BAD MAZE: Path between reward ports is too long (>25)")
        return False

    # Make sure all critical choice points are >=6 hexes away from a reward port
    choice_points = find_all_critical_choice_points(graph)
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

def single_barrier_moved(barriers_1, barriers_2):
    ''' Check if two sets of barriers differ by only one element. '''
    
    # The symmetric difference (XOR) between the sets must have exactly two elements
    # because each set should have exactly one barrier not present in the other set
    return len(barriers_1.symmetric_difference(barriers_2)) == 2


def have_common_path(paths1, paths2):
    '''
    Given 2 lists of hex paths, check if there is a common path between the 2 lists.
    Used for determining if there are shared optimal paths between mazes.
    
    Args:
    paths1 (list of lists): list of optimal hex paths between 2 reward ports
    paths2 (list of lists): list of optimal hex paths between 2 reward ports

    Returns:
    True if there is a common path between the 2 lists of paths, False otherwise.
    '''
    
    # Convert the path lists to tuples to make them hashable and store them in sets
    pathset1 = set(tuple(path) for path in paths1)
    pathset2 = set(tuple(path) for path in paths2)
    
    # Return True if there is 1 or more common path between the path sets, False otherwise
    return len(pathset1.intersection(pathset2)) > 0


def min_hex_diff_between_paths(paths1, paths2):
    '''
    Given 2 lists of hex paths, return the minimum number of hexes that differ 
    between the most similar paths in the 2 lists.
    Used for determining how different optimal paths are between mazes.
    
    Args:
    paths1 (list of lists): list of optimal hex paths between 2 reward ports
    paths2 (list of lists): list of optimal hex paths between 2 reward ports
    
    Returns:
    num_different_hexes (int): the min number of hexes different between a
    hex path in paths1 and a hex path in paths2. If there is 1 or more shared
    path between the path lists, the hex difference is 0.
    '''
    
    # If there is 1 or more shared path between the path sets, the hex difference is 0
    if have_common_path(paths1, paths2):
        return 0
    
    # Max possible number of different hexes between paths
    num_different_hexes = 25
    
    for patha in paths1:
        for pathb in paths2:
            # Get how many hexes differ between these paths
            diff = len(set(patha).symmetric_difference(set(pathb)))
            # Record the minimum possible difference between optimal paths
            if diff < num_different_hexes:
                num_different_hexes = diff
    
    return num_different_hexes


def have_common_optimal_paths(df, barriers_1, barriers_2):
    '''
    Given the hex maze database and 2 barrier sets, check if the 2 barrier sets have at
    least one common optimal path between every pair of reward ports (e.g. the barrier sets
    share an optimal path between ports 1 and 2, AND ports 1 and 3, AND ports 2 and 3), 
    meaning the rat could be running the same paths even though the barrier sets are different.
    
    (The result of this function is equivalent to checking if num_hexes_different_on_optimal_paths == 0)
    
    Returns:
    True if the barrier sets have a common optimal path between all pairs of reward ports, False otherwise
    '''
    # Do these barrier sets have a common optimal path from port 1 to port 2?
    have_common_path_12 = have_common_path(
        df_lookup(df, barriers_1, 'optimal_paths_12'), 
        df_lookup(df, barriers_2, 'optimal_paths_12'))
    # Do these barrier sets have a common optimal path from port 1 to port 3?
    have_common_path_13 = have_common_path(
        df_lookup(df, barriers_1, 'optimal_paths_13'), 
        df_lookup(df, barriers_2, 'optimal_paths_13'))
    # Do these barrier sets have a common optimal path from port 2 to port 3?
    have_common_path_23 = have_common_path(
        df_lookup(df, barriers_1, 'optimal_paths_23'), 
        df_lookup(df, barriers_2, 'optimal_paths_23'))
    
    # Return True if the barrier sets have a common optimal path between all pairs of reward ports
    return (have_common_path_12 and have_common_path_13 and have_common_path_23)


def num_hexes_different_on_optimal_paths(df, barriers_1, barriers_2):
    '''
    Given the hex maze database and 2 barrier sets, find the number of hexes different
    between optimal paths between every pair of reward ports. This helps us quantify
    how different two maze configurations are.
    
    Returns:
    num_different_hexes (int): the min number of hexes different in the most 
    similar optimal paths between all reward ports for the 2 mazes
    '''
    # How many hexes different are the most similar optimal paths from port 1 to port 2?
    num_hexes_different_12 = min_hex_diff_between_paths(
        df_lookup(df, barriers_1, 'optimal_paths_12'), 
        df_lookup(df, barriers_2, 'optimal_paths_12'))
    # How many hexes different are the most similar optimal paths from port 1 to port 3?
    num_hexes_different_13 = min_hex_diff_between_paths(
        df_lookup(df, barriers_1, 'optimal_paths_13'), 
        df_lookup(df, barriers_2, 'optimal_paths_13'))
    # How many hexes different are the most similar optimal paths from port 2 to port 3?
    num_hexes_different_23 = min_hex_diff_between_paths(
        df_lookup(df, barriers_1, 'optimal_paths_23'), 
        df_lookup(df, barriers_2, 'optimal_paths_23'))
    
    # Return the total number of hexes different between the most similar optimal
    # paths between all 3 reward ports
    return (num_hexes_different_12 + num_hexes_different_13 + num_hexes_different_23)


def at_least_one_path_shorter_and_longer(df, barriers_1, barriers_2):
    ''' 
    Given 2 sets of barriers, check if at least one optimal path between reward ports
    is shorter AND at least one is longer (e.g. the path length between ports 1 and 2
    increases and the path length between ports 2 and 3 decreases.
    
    Returns: 
    True if at least one path is shorter AND at least one is longer, False otherwise
    '''
    # Get path lengths between reward ports for each barrier set
    paths_1 = df_lookup(df, barriers_1, 'reward_path_lengths')
    paths_2 = df_lookup(df, barriers_2, 'reward_path_lengths')
    
    # Check if >=1 path is longer and >=1 path is shorter
    return (any(a < b for a, b in zip(paths_1, paths_2)) and any(a > b for a, b in zip(paths_1, paths_2)))


def optimal_path_order_changed(df, barriers_1, barriers_2):
    ''' 
    Given 2 sets of barriers, check if the length order of the optimal paths
    between reward ports has changed (e.g. the shortest path between reward ports
    used to be between ports 1 and 2 and is now between ports 2 and 3, etc.)
    
    Returns: 
    True if the optimal path length order has changed, False otherwise
    '''
    
    # Get path lengths between reward ports for each barrier set
    paths_1 = df_lookup(df, barriers_1, 'reward_path_lengths')
    paths_2 = df_lookup(df, barriers_2, 'reward_path_lengths')
    
     # Find which are the longest and shortest paths (multiple paths may tie for longest/shortest)
    longest_paths_1 = [i for i, num in enumerate(paths_1) if num == max(paths_1)]
    shortest_paths_1 = [i for i, num in enumerate(paths_1) if num == min(paths_1)]
    longest_paths_2 = [i for i, num in enumerate(paths_2) if num == max(paths_2)]
    shortest_paths_2 = [i for i, num in enumerate(paths_2) if num == min(paths_2)]
    
    # Check that both the longest and shortest paths are not the same
    return not any(l in longest_paths_2 and s in shortest_paths_2 for l in longest_paths_1 for s in shortest_paths_1)


def no_common_choice_points(df, barriers_1, barriers_2):
    ''' 
    Given 2 sets of barriers, ensure there are no common choice points between them.
    
    Returns: 
    True if there are no common choice points, False otherwise
    '''
    
    # Get the choice points for each barrier set
    choice_points_1 = df_lookup(df, barriers_1, 'choice_points')
    choice_points_2 = df_lookup(df, barriers_2, 'choice_points')
    
    # Check if there are no choice points in common
    return choice_points_1.isdisjoint(choice_points_2)


def get_barrier_change(barriers_1, barriers_2):
    '''
    Given 2 barrier sets that differ by the movement of a single barrier, 
    find the barrier that was moved.
    
    Args:
    barriers_1 (set/frozenset): The first barrier set
    barriers_2 (set/frozenset): The second barrier set
    
    Returns:
    old_barrier (int): The hex location of the barrier to be moved in the first set
    new_barrier (int): The hex location the barrier was moved to in the second set
    '''
    # Find the original barrier location
    old_barrier = barriers_1 - barriers_2

    # Find the new barrier location
    new_barrier = barriers_2 - barriers_1
    
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
    criteria_type='ANY': Accept new barrier sets that meet EITHER of these criteria
    criteria_type='ALL': (default) Accept new barrier sets that meet BOTH of these criteria
    
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
        criteria1 = at_least_one_path_shorter_and_longer(df, original_barriers, bar)
        # Check if the optimal path order has changed
        criteria2 = optimal_path_order_changed(df, original_barriers, bar)
        
        # Accept the potential new barrier set if it meets our criteria
        if criteria_type=='ALL':
            if (criteria1 and criteria2):
                bar = frozenset(int(b) for b in bar) # make int instead of np.int64
                new_barriers.append(bar)
        else: # If not specified as "ALL", I choose to assume "ANY"
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
        hex_diff = num_hexes_different_on_optimal_paths(df, original_barriers, barriers)
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
        next_sets = [s for s in next_sets if all(num_hexes_different_on_optimal_paths(df, s, v)>=min_hex_diff for v in visited)]
        
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


def get_barrier_sequence(df, start_barrier_set, min_hex_diff=8, max_sequence_length=5, max_recursive_calls=40):
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
        next_sets = get_next_barrier_sets(df, current_barrier_set, criteria_type="ANY")
        
        # Remove the current barrier set from the next sets to avoid self-referencing
        next_sets = [s for s in next_sets if s != current_barrier_set]
        
        # Remove barrier sets with optimal paths too similar to any other barrier set in the sequence
        next_sets = [s for s in next_sets if all(num_hexes_different_on_optimal_paths(df, s, v)>=min_hex_diff for v in visited)]
        
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


def get_isomorphic_mazes(barriers):
    '''
    Given a set of barriers defining a hex maze configuration, return the
    other 5 barrier sets that have the same graph structure (corresponding
    to the maze rotated clockwise/counterclockwise and reflected across its
    3 axes of symmetry)

    Args:
    barriers (set/frozenset): A set of barriers defining a hex maze
    
    Returns:
    set of frozensets: a set of the 5 barrier sets defining mazes isomorphic 
    to this maze
    '''
    # Rotate and reflect the maze to get other barrier configs that 
    # represent the same underlying graph structure
    reflected_ax1 = frozenset(get_reflected_barriers(barriers, axis=1))
    reflected_ax2 = frozenset(get_reflected_barriers(barriers, axis=2))
    reflected_ax3 = frozenset(get_reflected_barriers(barriers, axis=3))
    rotated_ccw = frozenset(get_rotated_barriers(barriers, direction='counterclockwise'))
    rotated_cw = frozenset(get_rotated_barriers(barriers, direction='clockwise'))
    
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
    The value of the attribute for this maze
    '''
    return df[(df['barriers'] == barriers)][attribute_name].item()


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
    choice_points = set(find_all_critical_choice_points(maze))
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


def get_barrier_sequence_attributes(df, barrier_sequence):
    '''
    Given the maze configuration database (df) and a sequence of 
    maze configurations that differ by the movement of a single barrier, 
    get the barrier change between each maze, reward path lengths and 
    choice points for all mazes in the sequence,
    and return a dictionary of these attributes.
    
    Args:
    df (dataframe): The database of all possible maze configurations.
    barrier_sequence (list of sets): The sequence of maze configurations.

    Returns:
    dict: A dictionary of attributes of this sequence.
    '''
    
    reward_path_lengths = []
    choice_points = []
    
    # Get attributes for each barrier set in the sequence
    for bars in barrier_sequence:
        reward_path_lengths.append(df_lookup(df, bars, 'reward_path_lengths'))
        choice_points.append(df_lookup(df, bars, 'choice_points'))
    
    barrier_changes = get_barrier_changes(barrier_sequence)
    
    # Set up a dictionary of attributes
    barrier_dict = {'barrier_sequence': barrier_sequence, 
                    'sequence_length': len(barrier_sequence),
                    'barrier_changes': barrier_changes,
                    'reward_path_lengths': reward_path_lengths,
                    'choice_points': choice_points}
    return barrier_dict


################################ Plotting hex mazes ################################

def plot_hex_maze(barriers, old_barrier=None, new_barrier=None):
    ''' 
    Given a set of barriers specifying a hex maze, plot the maze.
    Open hexes are shown in light blue, connected by thin grey lines.
    Barriers are shown in dark grey. Choice point(s) are in yellow.
    
    Option to specify old barrier location and new barrier location 
    to indicate a barrier change configuration:
    The now-open hex where the barrier used to be is shown in pale red.
    The new barrier is shown in dark red.
    
    Args:
    barriers (set): A set defining the hexes where barriers are placed in the maze.
    old_barrier (int): Optional. The hex where the barrier was in the previous maze.
    new_barrier (int): Optional. The hex where the new barrier is in this maze.
    '''
    
    # Create an empty maze for graph layout
    base_hex_maze = create_empty_hex_maze()
    
    # Create our actual maze
    maze = base_hex_maze.copy()
    for barrier in barriers:
        maze.remove_node(barrier)

    # Get the graph layout of the original maze
    pos = nx.kamada_kawai_layout(base_hex_maze)

    # Draw the available hexes in our maze using this layout
    nx.draw(maze, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=400)

    # Add the barriers in black
    nx.draw_networkx_nodes(base_hex_maze, pos, nodelist={b: b for b in barriers}, node_color='black', node_size=400)   
    nx.draw_networkx_labels(base_hex_maze, pos, labels={b: b for b in barriers}, font_color='white')

    # Make the choice point(s) yellow
    choice_points = find_all_critical_choice_points(maze)
    for choice_point in choice_points:
        nx.draw_networkx_nodes(base_hex_maze, pos, nodelist=[choice_point], node_color='yellow', node_size=400)
        
    # Make the old barrier location that is now an open hex light red
    if old_barrier is not None:
        nx.draw_networkx_nodes(base_hex_maze, pos, nodelist=[old_barrier], node_color='peachpuff', node_size=400)
    
    # Make the new barrier location dark red
    if new_barrier is not None:
        nx.draw_networkx_nodes(base_hex_maze, pos, nodelist=[new_barrier], node_color='darkred', node_size=400)
    
    plt.show()
    

def plot_barrier_change_sequence(barrier_sequence, print_barrier_info=True):
    '''
    Given a sequence of barrier sets that each differ by the movement of 
    a single barrier, plot each maze in the sequence with the moved barriers
    indicated on each maze.
    
    Open hexes are shown in light blue, connected by thin grey lines.
    Barriers are shown in dark grey. Choice point(s) are in yellow.
    The now-open hex where the barrier used to be is shown in pale red.
    The new barrier is shown in dark red.
    
    Args:
    barrier_sequence (list of sets): List of sequential barrier sets
    print_barrier_info (bool): Optional. Print each barrier set and the
    barrier moved between barrier sets. Defaults to True
    '''
    
    # Find the barriers moved from one configuration to the next
    barrier_changes = get_barrier_changes(barrier_sequence)
    
    # First print and plot the first barrier set
    if (print_barrier_info):
        print(f"Barrier set 0: {barrier_sequence[0]}")
    plot_hex_maze(barrier_sequence[0])
    
    # Now print barrier change info and plot each successive barrier set
    for i, (barriers, (old_barrier, new_barrier)) in enumerate(zip(barrier_sequence[1:], barrier_changes)):
        if (print_barrier_info):
            print(f"Barrier change: {old_barrier} -> {new_barrier}")
            print(f"Barrier set {i+1}: {barriers}")
        plot_hex_maze(barriers, old_barrier, new_barrier)


############## One-time use functions to help ensure that our database includes all possible mazes ##############
    
def num_isomorphic_mazes_in_set(set_of_valid_mazes, barriers):
    '''
    Given a set of all valid maze configurations and a set of barriers defining 
    a single hex maze configuration, find all isomorphic mazes for this 
    configuration that already exist in our larger set, and which are missing.

    Args:
    set_of_valid_mazes (set of frozensets): Set of all valid maze configurations
    barriers (set/frozenset): Set of barriers defining a single maze configuration
    
    Returns:
    int: The number of isomorphic mazes that already exist in the set
    list of sets: A list of isomorphic maze configurations missing from the set
    '''
    # Get all potential isomorphic mazes for this barrier configuration
    all_isomorphic_barriers = get_isomorphic_mazes(barriers)
    # Find other mazes in the dataframe that are isomorphic to the given barrier set
    isomorphic_barriers_in_set = set([b for b in set_of_valid_mazes if b in all_isomorphic_barriers])
    # Get the isomorphic mazes not present in the dataframe
    isomorphic_bariers_not_in_set = all_isomorphic_barriers.difference(isomorphic_barriers_in_set)
    return len(isomorphic_barriers_in_set), isomorphic_bariers_not_in_set


def num_isomorphic_mazes_in_df(df, barriers):
    '''
    Given our maze configuration database and a set of barriers defining 
    a hex maze configuration, find all isomorphic mazes that already exist 
    in the dataframe, and which are missing.

    Args:
    df (DataFrame): DataFrame containing all valid maze configurations in the
    column 'barriers'
    barriers (set/frozenset): Set of barriers defining a single maze configuration
    
    Returns:
    int: the number of isomorphic mazes that already exist in the dataframe
    list of sets: a list of isomorphic maze configurations missing from the dataframe
    '''
    # Get all potential isomorphic mazes for this barrier configuration
    all_isomorphic_barriers = get_isomorphic_mazes(barriers)
    # Find other mazes in the dataframe that are isomorphic to the given barrier set
    isomorphic_barriers_in_df = set([b for b in df['barriers'] if b in all_isomorphic_barriers])
    # Get the isomorphic mazes not present in the dataframe
    isomorphic_bariers_not_in_df = all_isomorphic_barriers.difference(isomorphic_barriers_in_df)
    return len(isomorphic_barriers_in_df), isomorphic_bariers_not_in_df
