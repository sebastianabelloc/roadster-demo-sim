import numpy as np
from roadster.quaternions import quaternion_conjugate, quaternion_multiplication


def attitude_error(q, q_ref):
    """Calculates the attitude error quaternion between the current attitude quaternion q and the reference quaternion q_ref.
    Checks for the sign of the scalar part to ensure the shortest rotation."""
    # convert input to numpy arrays
    q = np.asarray(q, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    # calculate attitude error using Hamilton product
    q_err = quaternion_multiplication(quaternion_conjugate(q_ref), q)
    # ensure scalar part is non-negative
    if q_err[0] < 0:
        q_err = -q_err
    return q_err


def pd_torque(q, omega, q_ref, omega_ref, kp, kd):
    """PD attitude control law: torque = -kp * q_err_vec - kd * (omega - omega_ref).
       q, q_ref are unit quaternions (w, x, y, z); omega, omega_ref in rad/s (body frame).
       Returns the commanded torque in N·m (body frame)."""
    # convert input to numpy arrays
    omega = np.asarray(omega, dtype=float)
    omega_ref = np.asarray(omega_ref, dtype=float)
    # attitude error (shortest path)
    q_err = attitude_error(q, q_ref)
    # P term pulls toward the reference, D term damps extra rotation
    torque = -kp * q_err[1:] - kd * (omega - omega_ref)
    return torque


def select_thrusters(torque_cmd, deadband):
    """Converts a commanded torque into on/off thruster pairs, axis by axis.
       Fires the positive or negative pair of each axis only if |torque| > deadband.
       Returns a boolean array of 12 thruster states."""
    on = np.zeros(12, dtype=bool)
    for axis in range(3):
        if torque_cmd[axis] > deadband:
            # positive pair of this axis
            on[4 * axis] = True
            on[4 * axis + 1] = True
        elif torque_cmd[axis] < -deadband:
            # negative pair of this axis
            on[4 * axis + 2] = True
            on[4 * axis + 3] = True
    return on


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # attitude error
    print("No error:      ", attitude_error([1, 0, 0, 0], [1, 0, 0, 0]), "expected [1, 0, 0, 0]")
    print("q vs -q:       ", attitude_error([1, 0, 0, 0], [-1, 0, 0, 0]), "expected [1, 0, 0, 0]")
    print("-90 deg on x:  ", attitude_error([0.707, -0.707, 0, 0], [1, 0, 0, 0]), "expected [0.707, -0.707, 0, 0]")

    # PD torque
    q_tilted = [np.cos(np.radians(-10)), np.sin(np.radians(-10)), 0, 0]  # -20 deg roll (half angle = -10 deg)
    torque = pd_torque(q_tilted, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], kp=1000, kd=0)
    print("PD torque, -20 deg roll:", torque, "expected ~[+174, 0, 0]")

    # thruster selection
    on = select_thrusters([500, -800, 50], deadband=100)
    print("Thrusters on:", np.where(on)[0], "expected [0 1 6 7]")