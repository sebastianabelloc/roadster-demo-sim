import numpy as np

def angular_acceleration(omega, inertia, torque):
    """Calculates the angular acceleration of the body given the angular velocity, torque, and inertia matrix."""
    # convert input to numpy arrays
    omega = np.asarray(omega, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    torque = np.asarray(torque, dtype=float)
    # calculate angular acceleration using Euler's equation: I * alpha = torque - omega x (I * omega)
    w_dot = np.linalg.inv(inertia) @ (np.subtract(torque, np.cross(omega, inertia @ omega)))
    return w_dot


if __name__ == "__main__":
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