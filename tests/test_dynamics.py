import numpy as np
from roadster.dynamics import angular_acceleration, state_derivative

# Box approximation of the Roadster
M, L, B, H = 2000.0, 4.2, 2.0, 1.2
INERTIA = np.diag([M / 12 * (B**2 + H**2),
                   M / 12 * (L**2 + H**2),
                   M / 12 * (L**2 + B**2)])
IXX, IYY, IZZ = np.diag(INERTIA)


def test_torque_on_x_no_spin():
    result = angular_acceleration([0, 0, 0], INERTIA, [IXX, 0, 0])
    np.testing.assert_allclose(result, [1, 0, 0])


def test_torque_on_y_no_spin():
    result = angular_acceleration([0, 0, 0], INERTIA, [0, IYY, 0])
    np.testing.assert_allclose(result, [0, 1, 0])


def test_spin_on_principal_axis_is_stable():
    # omega parallel to I*omega so the gyroscopic term must be 0
    result = angular_acceleration([1, 0, 0], INERTIA, [0, 0, 0])
    np.testing.assert_allclose(result, [0, 0, 0], atol=1e-12)


def test_gyroscopic_coupling():
    # omega = [1, 1, 0], no torque usign Euler gives w_dot_z = -(Iyy - Ixx) / Izz
    result = angular_acceleration([1, 1, 0], INERTIA, [0, 0, 0])
    expected = [0, 0, -(IYY - IXX) / IZZ]
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_state_derivative_free_flight():
    state0 = [0, 0, 0,   12, 0, 10,   1, 0, 0, 0,   0, 0, 0]
    result = state_derivative(0, state0, INERTIA, [0, 0, 0])
    expected = [12, 0, 10,   0, 0, -9.81,   0, 0, 0, 0,   0, 0, 0]
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_thruster_force_upright_pushes_up():
    # 19620 N up on 2000 kg cancels gravity
    state0 = [0, 0, 0,   0, 0, 0,   1, 0, 0, 0,   0, 0, 0]
    result = state_derivative(0, state0, INERTIA, [0, 0, 0], force=[0, 0, 19620], mass=2000.0)
    np.testing.assert_allclose(result[3:6], [0, 0, 0], atol=1e-12)


def test_thruster_force_upside_down_pushes_down():
    # rolled 180 deg, a force along the body z axis points to the ground: 2000 N on 2000 kg -> 1 m/s^2 more down
    state0 = [0, 0, 0,   0, 0, 0,   0, 1, 0, 0,   0, 0, 0]
    result = state_derivative(0, state0, INERTIA, [0, 0, 0], force=[0, 0, 2000], mass=2000.0)
    np.testing.assert_allclose(result[3:6], [0, 0, -9.81 - 1.0], atol=1e-12)

