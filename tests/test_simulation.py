import numpy as np
from roadster.simulation import simulate_roadster

M, L, B, H = 2000.0, 4.2, 2.0, 1.2
INERTIA = np.diag([M / 12 * (B**2 + H**2),
                   M / 12 * (L**2 + H**2),
                   M / 12 * (L**2 + B**2)])
G = 9.81
VX, VZ = 12.0, 10.0


def test_flight_time_matches_analytic():
    state0 = [0, 0, 0,   VX, 0, VZ,   1, 0, 0, 0,   0, 0, 0]
    sol = simulate_roadster(state0, INERTIA, [0, 0, 0])
    np.testing.assert_allclose(sol.t[-1], 2 * VZ / G, rtol=1e-6)


def test_range_matches_analytic():
    state0 = [0, 0, 0,   VX, 0, VZ,   1, 0, 0, 0,   0, 0, 0]
    sol = simulate_roadster(state0, INERTIA, [0, 0, 0])
    np.testing.assert_allclose(sol.y[0, -1], VX * 2 * VZ / G, rtol=1e-6)


def test_quaternion_norm_preserved_while_spinning():
    state0 = [0, 0, 0,   VX, 0, VZ,   1, 0, 0, 0,   0.3, 0.8, 0.1]
    sol = simulate_roadster(state0, INERTIA, [0, 0, 0])
    q_final = sol.y[6:10, -1]
    np.testing.assert_allclose(np.linalg.norm(q_final), 1.0, rtol=1e-6)