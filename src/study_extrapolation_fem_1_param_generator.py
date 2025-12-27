"""
Parameter Generator for Extrapolation Study.

Unlike the stochastic generator used in the training phase, this script provides
deterministic, predefined parameters based on a specific run index. It is designed
to be called by `_eval_02_plaxis_interaction.py`.

Functionality:
1. Generates a dense grid of distances between tunnels (0.5m to 10.0m).
2. Returns a JSON object with geometric parameters for a specific run index.
"""

import json
import sys
import numpy as np


def get_predefined_parameters(run_index: int):
    """
    Retrieves a fixed set of parameters for a specific simulation run.

    Configuration:
    - Tunnel geometry: Fixed at 3x3 meters.
    - Distance: Selected from a dense pre-calculated array based on the index.

    Args:
        run_index (int): The index of the current simulation run.

    Returns:
        dict: A dictionary containing the geometric parameters.
    """
    # 1. Generation of a dense distance array according to the evaluation plan
    distances_part1 = np.arange(0.5, 2.01, 0.1)  # Range 0.5 to 2.0 with step 0.1
    distances_part2 = np.arange(2.2, 3.01, 0.2)  # Range 2.2 to 3.0 with step 0.2
    distances_part3 = np.arange(3.5, 10.01, 0.5)  # Range 3.5 to 10.0 with step 0.5

    # Concatenate and round to avoid floating-point precision issues
    all_distances = np.round(np.concatenate([
        distances_part1, distances_part2, distances_part3
    ]), 2)

    if run_index >= len(all_distances):
        raise IndexError(f"Index {run_index} is out of bounds (total distances: {len(all_distances)}).")

    selected_distance = all_distances[run_index]

    # 2. Fixed geometric parameters
    parameters = {
        "distance": float(selected_distance),
        "vertical_shift": 0.0,
        "width_tunnel1": 3.0,
        "height_tunnel1": 3.0,
        "width_tunnel2": 3.0,
        "height_tunnel2": 3.0
    }
    return parameters


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Run index argument is missing. Usage: python _eval_01_param_generator.py <index>")
        sys.exit(1)

    try:
        index = int(sys.argv[1])
        params = get_predefined_parameters(index)
        # Output JSON to stdout for the parent process to read
        print(json.dumps(params))
    except (ValueError, IndexError) as e:
        print(f"Critical Error: {e}")
        sys.exit(1)