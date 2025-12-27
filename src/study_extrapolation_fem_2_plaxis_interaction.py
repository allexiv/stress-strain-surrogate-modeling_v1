"""
Plaxis Interaction Script for Extrapolation Evaluation.

This script manages the interaction with Plaxis 2D to perform a series of
Finite Element Method (FEM) calculations. It is adapted from the training
data generation script with specific modifications for the evaluation phase:
1. Executes a predefined series of 35 calculations based on distance variations.
2. Extracts plastic point history (Mohr-Coulomb) and maps it to mesh nodes.
3. Saves raw results to a dedicated evaluation directory.
"""

import subprocess
import os
import csv
import json
import time
import re
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from plxscripting.easy import *
import config

# --- CONFIGURATION ---
EVAL_RAW_DATA_DIR = Path(config.ROOT_DIR) / "data" / "evaluation_extrapolation" / "01_raw"
EVAL_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
NODE_RESULT_TYPES = config.RESULT_TYPES


def process_results(result):
    """
    Helper function to normalize the output from Plaxis API calls.
    Ensures the result is returned as a list or a single value.
    """
    if result is None: return []
    if isinstance(result, (int, float)): return [result]
    if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], list): return result[0]
    if hasattr(result, '__iter__'): return list(result)
    return []


def create_tunnel(center_x, center_y, width, height):
    """Constructs a tunnel geometry in the Plaxis model."""
    half_width = width / 2
    radius = height * (1 / 3)
    tunnel = g_i.tunnel(center_x, center_y)
    tunnel.CrossSection.add()
    tunnel.CrossSection.Segments[0].LineProperties.Delta1 = half_width
    tunnel.CrossSection.add()
    tunnel.CrossSection.Segments[1].LineProperties.Delta2 = (height - radius)
    tunnel.CrossSection.Segments[1].LineProperties.Delta1 = 0
    tunnel.CrossSection.add()
    tunnel.CrossSection.Segments[2].SegmentType = "Arc"
    tunnel.CrossSection.Segments[2].ArcProperties.Radius = radius
    tunnel.CrossSection.add()
    tunnel.CrossSection.Segments[3].LineProperties.Length = half_width - radius
    tunnel.CrossSection.symmetricclose()
    g_i.generatetunnel(tunnel)


# Initialize Plaxis Server Connection
s_i, g_i = new_server('localhost', config.INPUT_PORT, password=config.PASSWORD)

# --- EXECUTION LOOP CONFIGURATION ---
distances_part1 = np.arange(0.5, 2.01, 0.1)
distances_part2 = np.arange(2.2, 3.01, 0.2)
distances_part3 = np.arange(3.5, 10.01, 0.5)
all_distances = np.round(np.concatenate([distances_part1, distances_part2, distances_part3]), 2)
total_runs = len(all_distances)
print(f"--- Starting evaluation series of {total_runs} calculations ---")

for run_index, distance_val in enumerate(all_distances):
    print(f"\n--- Calculation {run_index + 1}/{total_runs} ---")

    # Define parameters for the current run
    parameters = {
        "distance": float(distance_val), "vertical_shift": 0.0,
        "width_tunnel1": 3.0, "height_tunnel1": 3.0,
        "width_tunnel2": 3.0, "height_tunnel2": 3.0
    }
    print(f"Parameters: Distance = {parameters['distance']:.2f} m")

    distance = parameters["distance"]
    vertical_shift = parameters["vertical_shift"]
    width_tunnel1 = parameters["width_tunnel1"]
    height_tunnel1 = parameters["height_tunnel1"]
    width_tunnel2 = parameters["width_tunnel2"]
    height_tunnel2 = parameters["height_tunnel2"]
    globals_dict = parameters.copy()

    # Start New Project
    s_i.new()

    # Geometry Definition
    g_i.SoilContour.initializerectangular(0, 0, config.MODEL_WIDTH, config.MODEL_HEIGHT)
    g_i.borehole(0)
    g_i.soillayer(g_i.Boreholes[0], 0)
    g_i.Soillayer_1.Zones[0].Bottom = -config.MODEL_HEIGHT

    # Material Assignment
    material = g_i.soilmat()
    for key, value in config.MATERIAL_PARAMS.items():
        setattr(material, key, value)
    g_i.setmaterial(g_i.Soillayers[0], material)

    # Structure Definition (Tunnels)
    g_i.gotostructures()
    distance_between_centers = distance + (width_tunnel1 + width_tunnel2) / 2
    k = 0.5
    center_x_1 = config.MODEL_WIDTH / 2 - k * distance_between_centers
    center_x_2 = config.MODEL_WIDTH / 2 + (1 - k) * distance_between_centers
    y_level_tunnel1 = -config.MODEL_HEIGHT / 2
    y_level_tunnel2 = -config.MODEL_HEIGHT / 2 + vertical_shift
    create_tunnel(center_x_1, y_level_tunnel1, width_tunnel1, height_tunnel1)
    create_tunnel(center_x_2, y_level_tunnel2, width_tunnel2, height_tunnel2)

    # Load Application
    line_load_line = g_i.line((0, 0), (config.MODEL_WIDTH, 0))[-1]
    line_load = g_i.lineload(line_load_line)
    line_load.qy_start = config.INITIAL_LINE_LOAD

    # Mesh Generation (with local refinement)
    g_i.gotomesh()
    g_i.Line_1_1.CoarsenessFactor = 1
    num_polycurves_per_tunnel = 8
    num_tunnels = 2
    for tunnel_index in range(num_tunnels):
        for polycurve_index in range(1, num_polycurves_per_tunnel + 1):
            polycurve_name = f"Polycurve_{tunnel_index * num_polycurves_per_tunnel + polycurve_index}_1"
            try:
                for _ in range(6):
                    g_i.refine(getattr(g_i, polycurve_name))
            except AttributeError:
                pass
    g_i.mesh(config.MESH_COARSENESS)

    # Calculation Phases
    g_i.gotostages()
    g_i.Line_1_1.activate(g_i.InitialPhase)
    phase1 = g_i.phase(g_i.InitialPhase)
    g_i.setcurrentphase(phase1)
    try:
        check_result = g_i.checkgeometry(11)
        unique_polygons = list(set(re.findall(r"BoreholePolygon_\d+_\d+", check_result)))
        for poly_name in unique_polygons:
            getattr(g_i, poly_name).deactivate(phase1)
    except Exception:
        pass
    g_i.calculate()
    time.sleep(1)

    # --- DATA EXTRACTION ---
    output_port = g_i.view(phase1)
    s_o, g_o = new_server('localhost', output_port, password=config.PASSWORD)
    initial_phase = g_o.Phases[0]
    last_phase = g_o.Phases[-1]

    # Step 2.1: Extract standard node results
    initial_results = {}
    last_results = {}
    print("Extracting node results...")
    for res_type in NODE_RESULT_TYPES:
        initial_results[res_type] = np.array(
            process_results(g_o.getresults(initial_phase, getattr(g_o.ResultTypes.Soil, res_type), "node")))
        last_results[res_type] = np.array(
            process_results(g_o.getresults(last_phase, getattr(g_o.ResultTypes.Soil, res_type), "node")))

    node_coords = np.vstack((initial_results['X'], initial_results['Y'])).T
    if node_coords.shape[0] == 0:
        print("Error: Failed to extract nodes. Skipping.")
        s_o.close()
        continue

    # Step 2.2: Extract plastic points (stress points)
    print("Extracting plastic points (stresspoint)...")
    sp_x = np.array(process_results(g_o.getresults(last_phase, g_o.ResultTypes.Soil.X, "stresspoint")))
    sp_y = np.array(process_results(g_o.getresults(last_phase, g_o.ResultTypes.Soil.Y, "stresspoint")))
    sp_plastic_flags = np.array(
        process_results(g_o.getresults(last_phase, g_o.ResultTypes.Soil.PlasticPointHistoryMohrCoulomb, "stresspoint")))

    # Step 2.3: Map stress points to nearest nodes
    is_plastic_for_nodes = np.zeros(len(node_coords))
    if sp_x.size > 0:
        print("Mapping stress points to nodes...")
        sp_coords = np.vstack((sp_x, sp_y)).T
        sp_tree = KDTree(sp_coords)
        _, nearest_sp_indices = sp_tree.query(node_coords, k=1)
        is_plastic_for_nodes = sp_plastic_flags[nearest_sp_indices]
    else:
        print("Warning: No plastic points found.")

    # Data Assembly
    node_data_list = []
    last_node_ids_set = set(last_results.get("NodeID", []))
    for i, node_id in enumerate(initial_results["NodeID"]):
        is_excavated = 1 if node_id not in last_node_ids_set else 0
        row_data = {"NodeID": node_id, "Excavated_soil": is_excavated, "is_plastic": is_plastic_for_nodes[i]}
        row_data.update(globals_dict)

        if not is_excavated:
            try:
                last_index = np.where(last_results["NodeID"] == node_id)[0][0]
                for res_type in NODE_RESULT_TYPES:
                    row_data[res_type] = last_results[res_type][last_index]
            except IndexError:
                for res_type in NODE_RESULT_TYPES: row_data[res_type] = 0
        else:
            for res_type in NODE_RESULT_TYPES: row_data[res_type] = 0

        # Overwrite X, Y with initial coordinates (undeformed)
        row_data['X'] = initial_results['X'][i]
        row_data['Y'] = initial_results['Y'][i]

        node_data_list.append(row_data)

    # Save to CSV
    output_filename = EVAL_RAW_DATA_DIR / f"eval_dist_{distance:.2f}m.csv"

    fieldnames = list(globals_dict.keys()) + ["Excavated_soil", "is_plastic"] + NODE_RESULT_TYPES

    if node_data_list:
        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(node_data_list)
        print(f"Results saved to: {output_filename}")

    s_o.close()
    time.sleep(1)

print("\nEvaluation calculation series completed.")
s_i.close()