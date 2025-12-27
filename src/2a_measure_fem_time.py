import subprocess
import time
import os
import json
import re
from plxscripting.easy import new_server
from plxscripting.plx_scripting_exceptions import PlxScriptingError
import config

# --- SETTINGS ---
N_ITERATIONS = 1
OUTPUT_FILE = "fem_calculation_time.json"


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


def run_parameter_generation():
    """Executes the subprocess to get random parameters."""
    script_path = os.path.join(os.path.dirname(__file__), config.SUBPROCESS_SCRIPT)
    try:
        process = subprocess.run(
            ["python", script_path],
            capture_output=True, text=True, check=True
        )
        return json.loads(process.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Error generating parameters: {e}")
        return None


def main():
    print(f"--- Starting FEM calculation time measurement ({N_ITERATIONS} iterations) ---")

    # Connect to Plaxis Input
    try:
        s_i, g_i = new_server('localhost', config.INPUT_PORT, password=config.PASSWORD)
    except PlxScriptingError as e:
        print(f"Connection error to Plaxis Input server: {e}")
        return

    calculation_times = []

    for i in range(1, N_ITERATIONS + 1):
        print(f"Iteration {i}/{N_ITERATIONS}...")

        # 1. Get parameters
        parameters = run_parameter_generation()
        if not parameters:
            print("  Skipping iteration due to parameter generation failure.")
            continue

        if not all(param in parameters for param in config.SUBPROCESS_PARAMS):
            print("  Missing required parameters. Skipping iteration.")
            continue

        # Extract parameters
        distance = parameters["distance"]
        vertical_shift = parameters["vertical_shift"]
        width_tunnel1 = parameters["width_tunnel1"]
        height_tunnel1 = parameters["height_tunnel1"]
        width_tunnel2 = parameters["width_tunnel2"]
        height_tunnel2 = parameters["height_tunnel2"]

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

        # Mesh Generation
        g_i.gotomesh()
        try:
            g_i.mesh(config.MESH_COARSENESS)
        except PlxScriptingError as e:
            print(f"  Mesh generation error: {e}. Skipping iteration.")
            continue

        # Stages & Geometry Check
        g_i.gotostages()
        phase1 = g_i.phase(g_i.InitialPhase)
        g_i.setcurrentphase(phase1)

        try:
            check_result = g_i.checkgeometry(11)
            polygon_pattern = r"BoreholePolygon_\d+_\d+"
            found_polygons = re.findall(polygon_pattern, check_result)
            for poly_name in set(found_polygons):
                getattr(g_i, poly_name).deactivate(phase1)
        except Exception:
            pass  # Ignore geometry check errors for deactivation logic

        # 3. CORE BLOCK: Measure Calculation Time
        try:
            print("  Starting calculation...")
            start_time = time.time()
            g_i.calculate()
            end_time = time.time()

            elapsed_time = end_time - start_time
            calculation_times.append(elapsed_time)
            print(f"  Calculation time: {elapsed_time:.2f} s")

        except PlxScriptingError as e:
            print(f"  Calculation error: {e}. Skipping iteration.")
            continue

    # 4. Process and Save Results
    if calculation_times:
        average_time = sum(calculation_times) / len(calculation_times)
        print(
            f"\nAverage FEM calculation time over {len(calculation_times)} successful iterations: {average_time:.2f} s")

        result_data = {"average_fem_calculation_time_s": average_time}
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(result_data, f, indent=4)
        print(f"Result saved to file: {OUTPUT_FILE}")
    else:
        print("\nNo successful iterations completed. Result not saved.")

    s_i.close()
    print("--- Measurement completed ---")


if __name__ == "__main__":
    main()