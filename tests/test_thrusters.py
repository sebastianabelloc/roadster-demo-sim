import numpy as np
from roadster.thrusters import thruster_torque, thruster_force, THRUSTER_DIRECTIONS

MAX = [10000, 10000, 10000, 10000, 3000, 3000]


def test_left_bottom_nozzles_roll_positive():
    throttle = [1, 0, 1, 0, 0, 0]
    np.testing.assert_allclose(thruster_torque(throttle, MAX), [15000, 0, 0])
    np.testing.assert_allclose(thruster_force(throttle, MAX), [0, 0, 20000])


def test_rear_bottom_nozzles_pitch_positive():
    throttle = [0, 0, 1, 1, 0, 0]
    np.testing.assert_allclose(thruster_torque(throttle, MAX), [0, 27000, 0])


def test_all_off_gives_zero_torque():
    np.testing.assert_allclose(thruster_torque([0] * 6, MAX), [0, 0, 0])


def test_no_nozzle_blows_up():
    # the car has no nozzles on the roof: no force can point down (gas blowing up)
    assert np.all(THRUSTER_DIRECTIONS[:, 2] >= 0)


def test_side_right_nozzle_gives_yaw_positive():
    torque = thruster_torque([0, 0, 0, 0, 1, 0], MAX)
    assert torque[2] > 0
