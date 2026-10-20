import numpy as np

# car dimensions
half_length = 2.1  # m
nozzle_x = 1.35   # m, bottom nozzles in front of / behind the center of mass
nozzle_y = 0.75   # m, bottom nozzles to the left / right of the center of mass
floor = -0.38     # m, height of the floor below the center of mass
half_width = 1.0  # m, side nozzles on the side of the car
wheel_z = -0.3    # m, height of the side nozzles (front wheels)

# thruster positions from center of mass (body frame: x forward, y left, z up)
THRUSTER_POSITIONS = np.array([
    [nozzle_x, nozzle_y, floor],    # 0 bottom front left
    [nozzle_x, -nozzle_y, floor],   # 1 bottom front right
    [-nozzle_x, nozzle_y, floor],   # 2 bottom rear left
    [-nozzle_x, -nozzle_y, floor],  # 3 bottom rear right
    [nozzle_x, -half_width, wheel_z],       # 4 side right (front wheel)
    [nozzle_x, half_width, wheel_z],      # 5 side left (front wheel)
], dtype=float)

# direction of the force each thruster applies on the car (body frame), opposite to where the gas goes
THRUSTER_DIRECTIONS = np.array([
    [0, 0, 1],    # 0 bottom front left: gas down
    [0, 0, 1],    # 1 bottom front right: gas down
    [0, 0, 1],    # 2 bottom rear left: gas down
    [0, 0, 1],    # 3 bottom rear right: gas down
    [0, 1, 0],    # 4 side right: gas to the right
    [0, -1, 0],    # 5 side left: gas to the left
], dtype=float)

def thruster_torque(throttle, max_thrust):
    """Calculates the torque on the car (body frame, N·m) given the throttle of each thruster (0 to 1)
       and the maximum thrust of each thruster in Newtons."""
    # convert input to numpy arrays
    throttle = np.asarray(throttle, dtype=float)
    max_thrust = np.asarray(max_thrust, dtype=float)
    # force vector of each thruster: direction times its thrust (6 x 3)
    forces = THRUSTER_DIRECTIONS * (throttle * max_thrust)[:, None]
    # torque of each thruster is position x force, then add them all up
    torque = np.cross(THRUSTER_POSITIONS, forces).sum(axis=0)
    return torque

def thruster_force(throttle, max_thrust):
    """Calculates the net force of the thrusters on the car (body frame, N) given the throttle of each thruster (0 to 1)
       and the maximum thrust of each thruster in Newtons. The nozzles do not cancel each other, so they also push the car."""
    # convert input to numpy arrays
    throttle = np.asarray(throttle, dtype=float)
    max_thrust = np.asarray(max_thrust, dtype=float)
    # force vector of each thruster by multiplying direction times its thrust, then adding them all up
    force = (THRUSTER_DIRECTIONS * (throttle * max_thrust)[:, None]).sum(axis=0)
    return force

if __name__ == "__main__":
    # both left bottom nozzles at full thrust: roll positive torque and an upward push
    throttle = [1, 0, 1, 0, 0, 0]
    max_thrust = [10000, 10000, 10000, 10000, 3000, 3000]  # N
    print("Thruster torque:", thruster_torque(throttle, max_thrust), " expected [15000, 0, 0]")
    print("Thruster force: ", thruster_force(throttle, max_thrust), " expected [0, 0, 20000]")