import numpy as np

# car dimensions
half_length = 2.1  # m
half_width = 1.0  # m

# thruster positions from center of mass (body frame)
THRUSTER_POSITIONS = np.array([
    [0, half_width, 0],     # roll positive
    [0, -half_width, 0],    # roll positive
    [0, half_width, 0],     # roll negative
    [0, -half_width, 0],    # roll negative
    [half_length, 0, 0],    # pitch positive
    [-half_length, 0, 0],   # pitch positive
    [half_length, 0, 0],    # pitch negative
    [-half_length, 0, 0],   # pitch negative
    [half_length, 0, 0],    # yaw positive
    [-half_length, 0, 0],   # yaw positive
    [half_length, 0, 0],    # yaw negative
    [-half_length, 0, 0],   # yaw negative
], dtype=float)

# direction of the force each thruster applies on the car (body frame)
THRUSTER_DIRECTIONS = np.array([
    [0, 0, 1],    # roll positive
    [0, 0, -1],   # roll positive
    [0, 0, -1],   # roll negative
    [0, 0, 1],    # roll negative
    [0, 0, -1],   # pitch positive
    [0, 0, 1],    # pitch positive
    [0, 0, 1],    # pitch negative
    [0, 0, -1],   # pitch negative
    [0, 1, 0],    # yaw positive
    [0, -1, 0],   # yaw positive
    [0, -1, 0],   # yaw negative
    [0, 1, 0],    # yaw negative
], dtype=float)

def thruster_torque(on, thrust):
    """Calculates the torque on the car given the thruster states and thrust magnitude.
       Accepts a boolean array of thruster states (on/off) and a thrust magnitude in Newtons."""
    # convert input to numpy arrays
    on = np.asarray(on, dtype=bool)
    thrust = float(thrust)
    torque = np.cross(THRUSTER_POSITIONS[on], THRUSTER_DIRECTIONS[on] * thrust).sum(axis=0)
    return torque

if __name__ == "__main__":
    # test thruster torque calculation
    on = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    thrust = 500.0  # N
    torque = thruster_torque(on, thrust)
    print("Thruster torque:", torque)