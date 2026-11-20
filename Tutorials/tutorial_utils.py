"""
tutorial_utils.py

Helpers for loading the example data used in the tutorials, and marking its blocks on plots.
"""

import json
import re

import pandas as pd

PORT_HEX = {"A": 1, "B": 2, "C": 3}


def load_session(csv_path):
    """
    Load a session file (e.g. example_data/IM-1871_20250802_ep0_98178d7d.csv) into the
    things a port or hex value learner needs.

    Each row of the file is one trial. The columns we use:
    - end_port: the port the rat ran to (A, B or C)
    - hex_path: the hexes the rat ran, in order
    - reward: 1 if the trial was rewarded, 0 if not
    - sweeps: the hexes the rat represented while at each hex of hex_path
    - port_representation: the hexes represented while the rat was at the port (we use this as SWR replay)
    The commented header at the top of the file has the maze and reward probabilities for each block,
    and whether it's a barrier change or probability change session.

    Parameters:
        csv_path (str): Path to the session file

    Returns:
        dict: ports, trajectories, rewards, swept_hexes, and replayed_hexes (one entry per
            trial), plus the maze and reward probabilities [pA, pB, pC] for each block,
            and how many trials were run in each block, and the session type
            ("barrier" or "probability" change)
    """
    # Get the maze and reward probabilities for each block from the header, and the
    # session type (each block line ends with e.g. "barrier change")
    mazes, reward_probs = [], []
    for line in open(csv_path):
        if not line.startswith("#"):
            break
        block = re.match(r"# block \d+: maze ([\d,]+);.*p\(A\)=([\d.]+), p\(B\)=([\d.]+), p\(C\)=([\d.]+)", line)
        if block:
            mazes.append({int(hex) for hex in block.group(1).split(",")})
            reward_probs.append([float(block.group(i)) / 100 for i in (2, 3, 4)])
            session_type = line.split(";")[-1].split()[0]

    trials = pd.read_csv(csv_path, comment="#")

    trajectories, swept_hexes = [], []
    for path, sweeps, start_port, end_port in zip(
        trials["hex_path"], trials["sweeps"], trials["start_port"], trials["end_port"]
    ):
        path = [int(hex) for hex in path.split()]
        sweeps = json.loads(sweeps)

        # The decoded path sometimes stops a hex short of the port, so make sure
        # each path starts and ends at its recorded ports (with no sweeps there)
        if start_port in PORT_HEX and path[0] != PORT_HEX[start_port]:
            path, sweeps = [PORT_HEX[start_port]] + path, [[]] + sweeps
        if path[-1] != PORT_HEX[end_port]:
            path, sweeps = path + [PORT_HEX[end_port]], sweeps + [[]]

        trajectories.append(path)
        swept_hexes.append(sweeps)

    return {
        "ports": trials["end_port"].tolist(),
        "trajectories": trajectories,
        "rewards": trials["reward"].astype(float).tolist(),
        "swept_hexes": swept_hexes,
        "replayed_hexes": [
            [int(hex) for hex in hexes.split()] for hexes in trials["port_representation"].fillna("")
        ],
        "mazes": mazes,
        "reward_probs": reward_probs,
        "trials_per_block": trials.groupby("block").size().tolist(),
        "session_type": session_type,
    }


def draw_blocks(ax, session, show_probabilities=True):
    """
    Draw a red line on a plot of trials where each block after the first begins (where the reward
    probabilities or the barriers changed), and (optionally) the reward probabilities above it whenever
    they change.

    Parameters:
        ax (matplotlib Axes): The plot, with trial number (starting at 1) on the x axis
        session (dict): A session from load_session
        show_probabilities (bool): Whether to write the reward probabilities above the plot
    """
    block_start = 1
    previous_probabilities = None
    for probabilities, n_trials in zip(session["reward_probs"], session["trials_per_block"]):
        if block_start > 1:
            ax.axvline(block_start - 0.5, color="red", linewidth=1)
        # Barrier change sessions keep the same reward probabilities, so only write them when they change
        if show_probabilities and probabilities != previous_probabilities:
            ax.text(block_start, 1.02, "   ".join(f"{name}: {p:.0%}" for name, p in zip("ABC", probabilities)),
                    fontsize=10, va="bottom")
        previous_probabilities = probabilities
        block_start += n_trials
