import subprocess
import time
import os
import csv
import json
import datetime
import re
from plxscripting.easy import new_server
import config


def create_tunnel(g_i, center_x, center_y, width, height):
    """Creates geometry for a single tunnel."""
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


def get_timestamped_filename(base_dir, base_filename="results"):
    """Generates a filename with the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{base_filename}_{timestamp}")


def run_parameter_generation():
    """Executes the subprocess to get random parameters."""
    script_path = os.path.join(os.path.dirname(__file__), config.SUBPROCESS_SCRIPT)
    try:
        process = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, check=True
        )
        return json.loads(process.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    # Connect to Plaxis Input and Output servers
    try:
        s_i, g_i = new_server('localhost', config.INPUT_PORT, password=config.PASSWORD)
        s_o, g_o = new_server('localhost', config.OUTPUT_PORT, password=config.PASSWORD)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    iteration_number = 0

    while True:
        iteration_number += 1
        print(f"Calculation {iteration_number}")

        # 1. Get parameters
        parameters = run_parameter_generation()

        if not parameters:
            print("Error executing subprocess or empty parameters. Ending loop.")
            break

        if not all(param in parameters for param in config.SUBPROCESS_PARAMS):
            print("Missing required parameters. Ending loop.")
            break

        # Extract parameters
        distance = parameters["distance"]
        vertical_shift = parameters["vertical_shift"]
        width_tunnel1 = parameters["width_tunnel1"]
        height_tunnel1 = parameters["height_tunnel1"]
        width_tunnel2 = parameters["width_tunnel2"]
        height_tunnel2 = parameters["height_tunnel2"]

        # Global parameters for CSV
        globals_dict = {
            "distance": distance,
            "vertical_shift": vertical_shift,
            "width_tunnel1": width_tunnel1,
            "height_tunnel1": height_tunnel1,
            "width_tunnel2": width_tunnel2,
            "height_tunnel2": height_tunnel2,
        }

        # 2. Build Model
        s_i.new()

        # Soil and Borehole
        g_i.SoilContour.initializerectangular(0, 0, config.MODEL_WIDTH, config.MODEL_HEIGHT)
        g_i.borehole(0)
        g_i.soillayer(g_i.Boreholes[0], 0)
        g_i.Soillayer_1.Zones[0].Bottom = -config.MODEL_HEIGHT

        # Material assignment
        material = g_i.soilmat()
        for key, value in config.MATERIAL_PARAMS.items():
            setattr(material, key, value)
        g_i.setmaterial(g_i.Soillayers[0], material)

        # Structures (Tunnels)
        g_i.gotostructures()
        distance_between_centers = distance + (width_tunnel1 + width_tunnel2) / 2
        k = 0.5
        center_x_1 = config.MODEL_WIDTH / 2 - k * distance_between_centers
        center_x_2 = config.MODEL_WIDTH / 2 + (1 - k) * distance_between_centers
        y_level_tunnel1 = -config.MODEL_HEIGHT / 2
        y_level_tunnel2 = -config.MODEL_HEIGHT / 2 + vertical_shift

        create_tunnel(g_i, center_x_1, y_level_tunnel1, width_tunnel1, height_tunnel1)
        create_tunnel(g_i, center_x_2, y_level_tunnel2, width_tunnel2, height_tunnel2)

        # Line Load
        line_load_line = g_i.line((0, 0), (config.MODEL_WIDTH, 0))[-1]
        line_load = g_i.lineload(line_load_line)
        line_load.qy_start = config.INITIAL_LINE_LOAD

        # Mesh Generation and Refinement
        g_i.gotomesh()
        g_i.Line_1_1.CoarsenessFactor = 1
        num_polycurves_per_tunnel = 8
        num_tunnels = 2

        for tunnel_index in range(num_tunnels):
            for polycurve_index in range(1, num_polycurves_per_tunnel + 1):
                # Construct name dynamically based on Plaxis naming conventions
                polycurve_name = f"Polycurve_{tunnel_index * num_polycurves_per_tunnel + polycurve_index}_1"
                try:
                    for _ in range(6):
                        g_i.refine(getattr(g_i, polycurve_name))
                except AttributeError:
                    pass

        g_i.mesh(config.MESH_COARSENESS)

        # Staged Construction
        g_i.gotostages()
        g_i.Line_1_1.activate(g_i.InitialPhase)
        phase1 = g_i.phase(g_i.InitialPhase)
        g_i.setcurrentphase(phase1)

        # Handle geometry check and soil deactivation inside tunnels
        try:
            check_result = g_i.checkgeometry(11)
            polygon_pattern = r"BoreholePolygon_\d+_\d+"
            found_polygons = re.findall(polygon_pattern, check_result)
            unique_polygons = list(set(found_polygons))
            for poly_name in unique_polygons:
                try:
                    getattr(g_i, poly_name).deactivate(phase1)
                except AttributeError:
                    pass
        except Exception:
            pass

        # Calculate
        g_i.calculate()

        # 3. Post-Processing (Output)
        # Reconnect to the specific output port for this project view
        output_port = g_i.view(phase1)
        s_o, g_o = new_server('localhost', output_port, password=config.PASSWORD)

        initial_phase = g_o.Phases[0]
        last_phase = g_o.Phases[-1]

        def get_results_for_phase(phase):
            results = {}
            for result_type in config.RESULT_TYPES:
                # Retrieve node results
                results[result_type] = g_o.getresults(
                    phase, getattr(g_o.ResultTypes.Soil, result_type), "node"
                )
            return results

        initial_results = get_results_for_phase(initial_phase)
        last_results = get_results_for_phase(last_phase)

        def process_result_values(result):
            if result is None:
                return []
            if isinstance(result, (int, float)):
                return [result]
            if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], list):
                return result[0]
            if hasattr(result, '__iter__'):
                return list(result)
            return []

        initial_results = {k: process_result_values(v) for k, v in initial_results.items()}
        last_results = {k: process_result_values(v) for k, v in last_results.items()}

        last_node_ids_set = set(last_results["NodeID"])

        # Collect data per node
        node_data_list = []
        for i, node_id in enumerate(initial_results["NodeID"]):
            # Determine if soil was excavated (node missing in last phase)
            is_excavated = 1 if node_id not in last_node_ids_set else 0

            # Base info + global parameters
            row_data = {
                "NodeID": node_id,
                "Excavated_soil": is_excavated,
                "X": initial_results["X"][i],
                "Y": initial_results["Y"][i],
            }
            row_data.update(globals_dict)

            # Fill result data
            if not is_excavated:
                try:
                    last_index = last_results["NodeID"].index(node_id)
                    for result_type in config.RESULT_TYPES:
                        if result_type not in ["X", "Y"]:
                            row_data[result_type] = last_results[result_type][last_index]
                except ValueError:
                    # Fallback if node not found despite check
                    for result_type in config.RESULT_TYPES:
                        if result_type not in ["X", "Y"]:
                            row_data[result_type] = 0
            else:
                # Zero out results for excavated nodes
                for result_type in config.RESULT_TYPES:
                    if result_type not in ["X", "Y"]:
                        row_data[result_type] = 0

            node_data_list.append(row_data)

        # Save to CSV
        timestamped_filename = get_timestamped_filename(config.raw_data_dir, base_filename="results")
        combined_csv_file = timestamped_filename + ".csv"

        if node_data_list:
            # Define column order: globals -> basic info -> results
            fieldnames = [
                "distance", "vertical_shift",
                "width_tunnel1", "height_tunnel1",
                "width_tunnel2", "height_tunnel2",
                "Excavated_soil", "X", "Y"
            ]
            fieldnames.extend([rt for rt in config.RESULT_TYPES if rt not in ["X", "Y"]])

            with open(combined_csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in node_data_list:
                    writer.writerow(row)

    print("Calculation loop completed.")
    s_i.close()
    s_o.close()


if __name__ == "__main__":
    main()