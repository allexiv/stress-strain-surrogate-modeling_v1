import random
import json

def generate_random_parameters():
    """Generates random geometric parameters for the tunnels."""
    distance = round(random.uniform(1, 8), 1)
    vertical_shift = round(random.uniform(-4, 4), 1)
    width_tunnel1 = round(random.uniform(2, 5), 1)
    height_tunnel1 = round(random.uniform(2, 3), 1)
    width_tunnel2 = round(random.uniform(2, 5), 1)
    height_tunnel2 = round(random.uniform(2, 3), 1)

    parameters = {
        "distance": distance,
        "vertical_shift": vertical_shift,
        "width_tunnel1": width_tunnel1,
        "height_tunnel1": height_tunnel1,
        "width_tunnel2": width_tunnel2,
        "height_tunnel2": height_tunnel2
    }
    return parameters

if __name__ == "__main__":
    # Generate parameters and output to stdout in JSON format
    params = generate_random_parameters()
    print(json.dumps(params))