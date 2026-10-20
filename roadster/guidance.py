import numpy as np


def roll_reference(t, t_start=0.48, duration=2):
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

def vertical_reference(z, t, t_hover_start, z_hover=0.5, a_brake=0.6 * 9.81, t_hover=0.3, v_land=0.6):
    """Returns vertical velocity reference (m/s) based on current altitude z (m), time t (s), and hover start time t_hover_start (s) of designed mission."""
    if t_hover_start is None:
        # descent: as fast as we can still stop at z_hover braking with a_brake (v^2 = 2 a d)
        return -np.sqrt(2 * a_brake * max(z - z_hover, 0.0))
    elif t - t_hover_start < t_hover:
        # hover: hold altitude
        return 0.0
    else:
        # landing: slow constant descent until touchdown (z axis points up, so descending is negative)
        return -v_land
    
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    for t in [0.0, 0.9, 2.0]:
        phi_ref, omega_ref = roll_reference(t)
        print(f"t = {t:.2f} s -> phi_ref = {phi_ref:.2f} rad, omega_ref = {omega_ref:.2f} rad/s")

    for phi in [0, -np.pi / 2, -np.pi, -2 * np.pi]:
        print(f"phi = {phi:.2f} rad -> q_ref = {roll_quaternion(phi)}")

    print(vertical_reference(4.5, 2.6, None))      # descending, 4 m above hover  -> expected ~ -6.86
    print(vertical_reference(0.5, 4.0, 3.8))       # hovering                     -> expected 0.0
    print(vertical_reference(0.5, 4.7, 3.8))       # landing                      -> expected -0.3