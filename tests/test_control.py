import numpy as np
from roadster.control import attitude_error, pd_torque, allocate_thrusters, lift_command
from roadster.thrusters import thruster_torque, thruster_force

G = 9.81


def test_attitude_error_zero_when_on_reference():
    np.testing.assert_allclose(attitude_error([1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])


def test_attitude_error_takes_shortest_path():
    # q and -q are the same attitude: no error, no extra turn after the barrel roll
    np.testing.assert_allclose(attitude_error([1, 0, 0, 0], [-1, 0, 0, 0]), [1, 0, 0, 0], atol=1e-12)


def test_pd_torque_pushes_back_toward_reference():
    # car rolled -20 deg, reference level: roll torque must be positive (back to level)
    half = np.radians(-10)
    q = [np.cos(half), np.sin(half), 0, 0]
    torque = pd_torque(q, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], kp=1000, kd=0)
    assert torque[0] > 0
    np.testing.assert_allclose(torque[1:], [0, 0], atol=1e-12)


def test_pd_torque_damping_only_acts_on_extra_rate():
    # spinning exactly as the guidance asks: no D torque
    torque = pd_torque([1, 0, 0, 0], [-3, 0, 0], [1, 0, 0, 0], [-3, 0, 0], kp=0, kd=500)
    np.testing.assert_allclose(torque, [0, 0, 0])


MAX = np.array([10000] * 4 + [3000] * 2)


def test_allocation_gives_the_commanded_force_and_torque():
    # hover force with roll and pitch torques: the 4 bottom nozzles must reproduce them exactly
    force_z, torque_cmd = 19620.0, [3000.0, -2000.0, 0.0]
    throttle = allocate_thrusters(force_z, torque_cmd, deadband=100, bottom_max=10000)
    np.testing.assert_allclose(thruster_force(throttle, MAX), [0, 0, force_z])
    np.testing.assert_allclose(thruster_torque(throttle, MAX)[:2], torque_cmd[:2])


def test_allocation_nozzles_only_push():
    # no upward force asked, only roll: the right side stays off and the left side pushes
    throttle = allocate_thrusters(0.0, [3000.0, 0.0, 0.0], deadband=100, bottom_max=10000)
    assert np.all(throttle >= 0)
    np.testing.assert_allclose(throttle[[1, 3]], [0, 0])
    np.testing.assert_allclose(thruster_torque(throttle, MAX)[0], 3000.0)


def test_allocation_yaw_side_nozzles_and_deadband():
    assert np.all(allocate_thrusters(0.0, [0, 0, 50], 100, 10000)[4:] == 0)       # inside the deadband
    np.testing.assert_allclose(allocate_thrusters(0.0, [0, 0, 500], 100, 10000)[4:], [1, 0])
    np.testing.assert_allclose(allocate_thrusters(0.0, [0, 0, -500], 100, 10000)[4:], [0, 1])


def test_lift_command_cases():
    np.testing.assert_allclose(lift_command(-6.9, -6.9, 0.6 * G, kv=8.0), 1.6)   # on the braking curve
    np.testing.assert_allclose(lift_command(0.0, 0.0, 0.0, kv=8.0), 1.0)         # hovering: holds the weight
    np.testing.assert_allclose(lift_command(-3.0, -1.0, 0.0, kv=8.0), 1.9)       # saturates at the max
    np.testing.assert_allclose(lift_command(5.0, 0.0, 0.0, kv=8.0), 0.0)         # cannot pull down
