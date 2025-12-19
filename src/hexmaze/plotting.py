"""
plotting.py

This module contains functions for plotting hex mazes, plotting barrier sequences,
and working with hex centroids.
"""

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from itertools import chain
import networkx as nx
import numpy as np
import math

# For type hints
from typing import (
    Optional,
    Union,
    Mapping,
    Sequence,
    Literal,
)

from .utils import (
    create_empty_hex_maze, 
    maze_to_barrier_set,
    rotate_hex,
)
from .core import (
    get_optimal_paths_between_ports, 
    get_critical_choice_points, 
    get_maze_attributes, 
    get_optimal_paths
)
from .barrier_shift import (
    get_barrier_changes, 
    hexes_different_on_optimal_paths, 
    num_hexes_different_on_optimal_paths, 
    hexes_different_between_paths,
)

# Define the public interface for this module
__all__ = [
    "get_distance_to_nearest_neighbor", 
    "get_hex_sizes_from_centroids",
    "get_hex_centroids",
    "classify_triangle_vertices",
    "scale_triangle_from_centroid",
    "plot_hex_maze",
    "plot_barrier_change_sequence",
    "plot_hex_maze_comparison",
    "plot_hex_maze_path_comparison",
    "plot_evaluate_maze_sequence",
]


def get_distance_to_nearest_neighbor(hex_centroids: dict[int, tuple]) -> dict[int, float]:
    """
    Given a dictionary of hex: (x,y) centroid, calculate the minimum 
    euclidean distance to the closest neighboring hex centroid for each hex.

    Parameters:
        hex_centroids (dict): Dictionary of hex: (x, y) coords of centroid

    Returns:
        min_distances (dict): Dictionary of hex: minimum distance to the nearest hex
    """
    hex_ids = list(hex_centroids.keys())
    hex_coords = np.array(list(hex_centroids.values()))

    # Compute full pairwise distance matrix for all coordinates
    diffs = hex_coords[:, np.newaxis, :] - hex_coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))

    # Set distance between a hex and itself to infinity to exclude it
    np.fill_diagonal(distances, np.inf)

    # Get the minimum distance to a neighbor for each hex
    min_distances = {
        hex_id: float(np.min(dist_row))
        for hex_id, dist_row in zip(hex_ids, distances)
    }
    return min_distances


def get_hex_sizes_from_centroids(hex_centroids: dict[int, tuple]) -> dict:
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


def get_min_max_centroids(hex_centroids: dict[int, tuple]) -> tuple[float, float, float, float]:
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


def get_hex_centroids(view_angle:Literal[1, 2, 3]=1, scale:float=1, shift=[0, 0]) -> dict[int, tuple]:
    """
    Calculate the (x,y) coordinates of each hex centroid.
    Centroids are calculated relative to the centroid of the topmost hex at (0,0).

    Parameters:
        view_angle (int: 1, 2, or 3): The hex that is on the top point of the triangle
            when viewing the hex maze. Defaults to 1
        scale (float): The width of each hex (aka the length of the long diagonal,
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


def classify_triangle_vertices(vertices: list[tuple]) -> dict[str, tuple]:
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
    hex_positions:dict[int, tuple], 
    scale:float=1, 
    chop_vertices:bool=True, 
    chop_vertices_2:bool=False, 
    show_edge_barriers:bool=True
) -> list[tuple]:
    """
    Calculate the coordinates of the vertices of the base triangle that
    surrounds all of the hexes in the maze.
    Used as an easy way to show permanent barriers when plotting the hex maze.

    Parameters:
        hex_positions (dict): A dictionary of hex: (x,y) coordinates of centroids.
        scale (float): The width of each hex (aka the length of the long diagonal,
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


def get_stats_coords(hex_centroids: dict[int, tuple]) -> dict[str, tuple]:
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
        barriers = None,
        old_barrier: Optional[int] = None,
        new_barrier: Optional[int] = None,
        show_barriers: bool = True,
        show_choice_points: bool = True,
        show_optimal_paths: bool = False,
        show_arrow: bool = True,
        show_barrier_change: bool = True,
        show_hex_labels: bool = True,
        show_stats: bool = True,
        reward_probabilities: Optional[Sequence[float]] = None,
        show_permanent_barriers: bool = False,
        show_edge_barriers: bool = True,
        centroids: Optional[Mapping[int, tuple[float, float]]] = None,
        view_angle: Literal[1, 2, 3] = 1,
        hex_path: Optional[Sequence[int]] = None,
        arrows: Optional[Mapping[int, Sequence[int]]] = None,
        highlight_hexes: Optional[Union[set[int], Sequence[set[int]]]] = None,
        highlight_colors: Optional[Union[str, Sequence[str]]] = None,
        color_by: Optional[Mapping[int, float]] = None,
        colormap: Union[str, matplotlib.colors.Colormap] = "plasma",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        scale: float = 1.0,
        shift: Sequence[float] = (0.0, 0.0),
        ax: Optional[matplotlib.axes.Axes] = None,
        invert_yaxis: bool = False,
    ) -> None:
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
        hex_path (list[int]): List of hexes specifying a path taken through the maze.
            The path will be shown by black arrows between hexes. Defaults to None
        arrows (dict[int, list[int]]): Dictionary mapping source hex to one or more target hexes.
            Gray arrows will be drawn from each source hex to all of its target hexes. Defaults to None
        highlight_hexes (set[int] or list[set]): A set (or list[set]) of hexes to highlight.
            Takes precedence over other hex highlights (choice points, etc). Defaults to None.
        highlight_colors (string or list[string]): Color (or list[colors]) to highlight highlight_hexes.
            Each color in this list applies to the respective set of hexes in highlight_hexes.
            Defaults to 'darkorange' for a single group.
        color_by (dict): Dictionary of hex_id: value. Hexes will be colored by their corresponding 
            value using the specified colormap. Overridden by highlight_hexes. Defaults to None.
        colormap (str or matplotlib.colors.Colormap): Matplotlib colormap to use for 
            `color_by` values. Defaults to 'plasma'.
        vmin (float): Minimum value for colormap normalization, if using `color_by`. 
            Values in `color_by` less than `vmin` will be clipped to the lowest color. 
            If None, the minimum of `color_by` values is used. Defaults to None.
        vmax (float): Maximum value for colormap normalization, if using `color_by`.
            Values in `color_by` greater than `vmax` will be clipped to the highest color. 
            If None, the maximum of `color_by` values is used. Defaults to None.
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

    # If no axis was provided, create a new figure and axis to use
    if ax is None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        show_plot = False

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

    # Optional - Color hexes by values with a colormap
    if color_by is not None:
        # Ensure we only color existing hexes
        values = {hex: val for hex, val in color_by.items() if hex in hex_colors}

        if len(values) > 0:
            # Get colormap bounds from vmin/vmax or min/max of values
            actual_vmin = vmin if vmin is not None else min(values.values())
            actual_vmax = vmax if vmax is not None else max(values.values())

            # Get the colormap and normalize values to 0-1 range
            cmap = plt.get_cmap(colormap)
            norm = plt.Normalize(vmin=actual_vmin, vmax=actual_vmax)

            # Color hexes by value
            for h, val in values.items():
                hex_colors[h] = cmap(norm(val))

    # Optional – draw grey arrows between arbitrary hexes
    if arrows is not None:
        for source_hex, target_hexes in arrows.items():
            if source_hex not in hex_coordinates:
                continue

            start = np.array(hex_coordinates[source_hex])

            # Draw arrows from the source hex to all of its target hexes
            for target_hex in target_hexes:
                if target_hex not in hex_coordinates:
                    continue
                end = np.array(hex_coordinates[target_hex])

                ax.annotate(
                    "",
                    xy=end,
                    xycoords="data",
                    xytext=start,
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="grey",
                        linewidth=1,
                        shrinkA=scale * 0.35,
                        shrinkB=scale * 0.35,
                    ),
                )

    # Optional - plot a path through the maze with arrows
    if hex_path is not None and len(hex_path) > 1:
        # Set up counter for the number of each edge between hexes 
        # so we can offset the hex path arrows when the rat backtracks
        directed_edge_counts = defaultdict(int)
        undirected_edge_counts = defaultdict(int)

        for start_hex, end_hex in zip(hex_path[:-1], hex_path[1:]):
            # Skip hexes without coords (note that we do this before we remove barriers, 
            # because we may want to indicate a path that moves through a blocked hex)
            if start_hex not in hex_coordinates or end_hex not in hex_coordinates:
                continue

            start_coords = np.array(hex_coordinates[start_hex])
            end_coords = np.array(hex_coordinates[end_hex])

            # Count the number of times we've seen this hex transition
            edge_key = tuple((start_hex, end_hex))
            undirected_edge_key = tuple(sorted((start_hex, end_hex)))
            directed_edge_counts[edge_key] += 1
            undirected_edge_counts[undirected_edge_key] += 1

            # We count directed and undirected edges separately. Normally we want to offset based
            # on the directed edge so all arrows in a single direction are offset to the same side.
            # BUT we need to count undirected edges for the first backtrack, so a single reversal
            # such as 1->4->1 doesn't result in the 1->4 and 4->1 arrows on top of each other.
            if directed_edge_counts[edge_key] == 1 and undirected_edge_counts[undirected_edge_key] == 2:
                directed_edge_counts[edge_key] = 2 # bump the count for the first backtrack

            # Get arrow direction so we can offset the arrow for repeated edges
            direction = end_coords - start_coords
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            perpendicular_unit_vector = np.array([-direction[1], direction[0]]) / length
            offset = perpendicular_unit_vector  * 0.15 * scale * (directed_edge_counts[edge_key] - 1)

            ax.annotate(
                "",
                xy=end_coords + offset,
                xycoords="data",
                xytext=start_coords + offset,
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color='k',
                    linewidth=1.5,
                ),
            )

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
