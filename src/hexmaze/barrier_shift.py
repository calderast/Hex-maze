"""
barrier_shift.py

This module contains functions for comparing different maze configurations 
(especially with respect to quantifying how different two maze configurations are) 
and generating optimal barrier change sequences.
"""

import numpy as np
import pandas as pd
from .utils import (
    maze_to_barrier_set,
    get_rotated_barriers,
    get_reflected_barriers,
    get_isomorphic_mazes,
)
from .core import (
    get_critical_choice_points,
    get_optimal_paths,
    get_reward_path_lengths,
)

# Define the public interface for this module
__all__ = [
    "single_barrier_moved", 
    "have_common_path", 
    "have_common_optimal_paths",
    "min_hex_diff_between_paths",
    "hexes_different_between_paths",
    "hexes_different_on_optimal_paths",
    "num_hexes_different_on_optimal_paths",
    "num_hexes_different_on_optimal_paths_isomorphic",
    "at_least_one_path_shorter_and_longer",
    "optimal_path_order_changed",
    "no_common_choice_points",
    "get_barrier_change",
    "get_barrier_changes",
    "get_next_barrier_sets",
    "get_best_next_barrier_set",
    "find_all_valid_barrier_sequences",
    "get_barrier_sequence",
]


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
        criteria3 = set(get_reward_path_lengths(bar)) == {17, 19, 21}
        # Only 1 critical choice point
        criteria4 = len(get_critical_choice_points(bar)) == 1

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
