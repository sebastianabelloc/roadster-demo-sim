import numpy as np
from roadster.thrusters import thruster_torque, THRUSTER_DIRECTIONS


def test_pitch_negative_pair():
    on = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    np.testing.assert_allclose(thruster_torque(on, 500.0), [0, -2100, 0])


def test_all_off_gives_zero_torque():
    np.testing.assert_allclose(thruster_torque([0] * 12, 500.0), [0, 0, 0])


def test_each_pair_has_zero_net_force():
    # thrusters fire in pairs (0-1, 2-3, ..., 10-11): their forces must cancel
    for i in range(0, 12, 2):
        net_force = THRUSTER_DIRECTIONS[i] + THRUSTER_DIRECTIONS[i + 1]
        np.testing.assert_allclose(net_force, [0, 0, 0])