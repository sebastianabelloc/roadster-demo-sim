"""Run the closed-loop simulation and save the trajectory that Blender animates.

Usage (from the repo root):
    python scripts/export_trajectory.py renders/trajectory.npz

The file holds exactly what roadster.simulation returns (plus the kinematic ground roll):
    t          time [s], N samples
    states     13 x N  [r(3), v(3), q(4) (w, x, y, z), omega(3)]
    thrusters  12 x N  on/off of each attitude thruster
    lifts      N       lift command (fraction of the weight carried by the lift thrusters)
Blender reads nothing else about the motion.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roadster.simulation import simulate_controlled, add_ground_roll  # noqa: E402

# Box approximation of the Roadster: m = 2000 kg, L = 4.2 m, B = 2.0 m, H = 1.2 m
m, L, B, H = 2000.0, 4.2, 2.0, 1.2
I = np.diag([m / 12 * (B**2 + H**2), m / 12 * (L**2 + H**2), m / 12 * (L**2 + B**2)])

state0 = [0, 0, 0,   6.3, 0, 0.0,   1, 0, 0, 0,   0.1, 0.25, 0.05]
t, states, thrusters, lifts = simulate_controlled(state0, I, thrust=10000, kp=5e4, kd=3e3,
                                                  deadband=400, lift_flight=0.0, v_launch=7.0)
t_touchdown = t[-1]
t, states, thrusters, lifts = add_ground_roll(t, states, thrusters, lifts)

out = sys.argv[1] if len(sys.argv) > 1 else "renders/trajectory.npz"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
np.savez(out, t=t, states=states, thrusters=thrusters, lifts=lifts, t_takeoff=0.0, t_touchdown=t_touchdown)
print(f"saved {len(t)} samples from t = {t[0]:.2f} s to {t[-1]:.2f} s (touchdown at {t_touchdown:.2f} s) -> {out}")
