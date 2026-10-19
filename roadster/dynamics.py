import numpy as np
from roadster.quaternions import quaternion_derivative

GRAVITY = np.array([0.0, 0.0, -9.81])  # m/s^2, inertial frame (z up)

def angular_acceleration(omega, inertia, torque):
    """Calculates the angular acceleration of the body given the angular velocity, torque, and inertia matrix."""
    # convert input to numpy arrays
    omega = np.asarray(omega, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    torque = np.asarray(torque, dtype=float)
    # calculate angular acceleration using Euler's equation: I * alpha = torque - omega x (I * omega)
    w_dot = np.linalg.inv(inertia) @ (np.subtract(torque, np.cross(omega, inertia @ omega)))
    return w_dot

def state_derivative(t, state, inertia, torque, lift=0.0):
    """Calculates the derivative of the state vector given time t, state vector, inertia matrix, and torque vector.
       The state vector is [rx, ry, rz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz] where q is the quaternion and w is the angular velocity.
       Lift term is introduced to represent the fraction of the weight which is sustained by the thrusters. By default lift=0 simulates freefall"""
    # convert input to numpy arrays
    state = np.asarray(state, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    torque = np.asarray(torque, dtype=float)
    # isolate position, velocity, quaternion and angular velocity from state vector
    r = state[0:3]
    v = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    #calculate position derivative (velocity)
    r_dot = v
    # calculate velocity derivative assuming only gravity acts on the vehicle in the z direction
    v_dot = GRAVITY * (1 - lift)
    # calculate quaternion derivative
    q_dot = quaternion_derivative(q, omega)
    # calculate angular acceleration
    w_dot = angular_acceleration(omega, inertia, torque)
    # concatenate derivatives into a single array
    return np.concatenate((r_dot, v_dot, q_dot, w_dot))

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    # Box approximation of the Roadster: m = 2000 kg, L = 4.2 m, B = 2.0 m, H = 1.2 m
    m, L, B, H = 2000.0, 4.2, 2.0, 1.2
    I = np.diag([m / 12 * (B**2 + H**2),
                 m / 12 * (L**2 + H**2),
                 m / 12 * (L**2 + B**2)])
    print("Inertia diagonal:", np.diag(I))
    print("Case 1: torque on x, no spin      ->", angular_acceleration([0, 0, 0], I, [907, 0, 0]),  "expected ~[1, 0, 0]")
    print("Case 2: torque on y, no spin      ->", angular_acceleration([0, 0, 0], I, [0, 3180, 0]), "expected ~[0, 1, 0]")
    print("Case 3: spin on x, no torque      ->", angular_acceleration([1, 0, 0], I, [0, 0, 0]),    "expected [0, 0, 0]")
    print("Case 4: spin on x and y, no torque->", angular_acceleration([1, 1, 0], I, [0, 0, 0]),    "expected ~[0, 0, -0.63]")
    state0 = [0, 0, 0,   12, 0, 10,   1, 0, 0, 0,   0, 0, 0]
    d = state_derivative(0, state0, I, [0, 0, 0])
    print("State derivative:")
    print("  r_dot =", d[0:3],   " expected [12, 0, 10]")
    print("  v_dot =", d[3:6],   " expected [0, 0, -9.81]")
    print("  q_dot =", d[6:10],  " expected [0, 0, 0, 0]")
    print("  w_dot =", d[10:13], " expected [0, 0, 0]")