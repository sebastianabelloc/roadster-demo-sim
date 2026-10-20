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


def allocate_thrusters(force_z, torque_cmd, deadband, bottom_max, x_arm=1.35, y_arm=0.75):
    """Control allocation: shares the commanded upward force (N, body z) and torque (N·m, body frame) among the 6 thrusters,
       like the motors of a drone. The 4 bottom nozzles are throttled (0 to 1): all together give the upward force,
       more thrust on one side gives roll, more at the front or rear gives pitch. The 2 side nozzles are on/off for yaw.
       Returns an array of 6 throttles: bottom front left, front right, rear left, rear right, side right, side left."""
    # convert input to numpy array
    torque_cmd = np.asarray(torque_cmd, dtype=float)
    # sign of the position of each bottom nozzle: front (+) / rear (-) and left (+) / right (-)
    sx = np.array([1, 1, -1, -1])
    sy = np.array([1, -1, 1, -1])
    # share the roll and pitch torques among the 4 nozzles (roll = y * F, pitch = -x * F)
    attitude = torque_cmd[0] * sy / (4 * y_arm) - torque_cmd[1] * sx / (4 * x_arm)
    # common part: what the upward force asks for, but never less than needed to keep every nozzle pushing (>= 0)
    base = max(force_z / 4, -attitude.min(), 0.0)
    # thrust of each bottom nozzle as a throttle between 0 and 1
    bottom = np.clip((base + attitude) / bottom_max, 0.0, 1.0)
    # yaw with the side nozzles, on/off with a deadband
    side = np.zeros(2)
    if torque_cmd[2] > deadband:
        # right nozzle: gas to the right, nose turns left
        side[0] = 1.0
    elif torque_cmd[2] < -deadband:
        # left nozzle: gas to the left, nose turns right
        side[1] = 1.0
    return np.concatenate((bottom, side))

def lift_command(vz, vz_ref, a_ff, kv, lift_max=1.9, g=9.81):
    """Calculates the lift command based on current vertical velocity vz (m/s), reference vertical velocity vz_ref (m/s)."""
    # calculate lift command (fraction of the weight)
    lift_cmd = 1 + (a_ff + kv * (vz_ref - vz)) / g
    # thrusters cannot pull down or exceed their maximum
    lift_cmd = np.clip(lift_cmd, 0.0, lift_max)
    return lift_cmd

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

    # control allocation: hover force plus a roll torque -> more thrust on the left nozzles, no yaw
    throttle = allocate_thrusters(19620, [3000, 0, 50], deadband=100, bottom_max=10000)
    print("Throttles:", throttle, "expected [0.59 0.39 0.59 0.39 0 0]")

    # lift command (fraction of the weight held by the lift thrusters)
    g = 9.81
    print("On the braking curve:   ", lift_command(vz=-6.9, vz_ref=-6.9, a_ff=0.6 * g, kv=8.0), "expected 1.6")
    print("Hovering still:         ", lift_command(vz=0.0, vz_ref=0.0, a_ff=0.0, kv=8.0), "expected 1.0")
    print("Falling too fast:       ", lift_command(vz=-3.0, vz_ref=-1.0, a_ff=0.0, kv=8.0), "expected 1.9 (saturated)")
    print("Going up while hovering:", lift_command(vz=0.5, vz_ref=0.0, a_ff=0.0, kv=8.0), "expected ~0.59")