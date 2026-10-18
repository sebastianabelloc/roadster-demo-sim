import numpy as np


def roll_reference(t, t_start=0.45, duration=0.9):
    """Returns the reference roll angle (rad) and roll rate (rad/s) for the barrel roll.
       Smooth profile from 0 to -2*pi (driver side down), with zero rate at start and end."""
    if t < t_start:
        # before the roll: level and not rotating
        phi_ref = 0.0
        omega_ref = 0.0
    elif t < t_start + duration:
        # during the roll: smooth profile from 0 to -2*pi
        s = (t - t_start) / duration  # progress from 0 to 1
        phi_ref = -np.pi * (1 - np.cos(np.pi * s))
        omega_ref = -(np.pi**2 / duration) * np.sin(np.pi * s)
    else:
        # after the roll: full turn completed, not rotating
        phi_ref = -2 * np.pi
        omega_ref = 0.0
    return phi_ref, omega_ref

def roll_quaternion(phi):
    """Returns the quaternion corresponding to a roll angle phi (rad)."""
    half_phi = phi / 2
    return np.array([np.cos(half_phi), np.sin(half_phi), 0, 0])  # [w, x, y, z]

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    for t in [0.0, 0.9, 2.0]:
        phi_ref, omega_ref = roll_reference(t)
        print(f"t = {t:.2f} s -> phi_ref = {phi_ref:.2f} rad, omega_ref = {omega_ref:.2f} rad/s")

    for phi in [0, -np.pi / 2, -np.pi, -2 * np.pi]:
        print(f"phi = {phi:.2f} rad -> q_ref = {roll_quaternion(phi)}")