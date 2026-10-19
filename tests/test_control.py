import numpy as np
from roadster.control import attitude_error, pd_torque, select_thrusters, lift_command

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


def test_select_thrusters_pairs_and_deadband():
    on = select_thrusters([500, -800, 50], deadband=100)
    np.testing.assert_array_equal(np.where(on)[0], [0, 1, 6, 7])


def test_lift_command_cases():
    np.testing.assert_allclose(lift_command(-6.9, -6.9, 0.6 * G, kv=8.0), 1.6)   # on the braking curve
    np.testing.assert_allclose(lift_command(0.0, 0.0, 0.0, kv=8.0), 1.0)         # hovering: holds the weight
    np.testing.assert_allclose(lift_command(-3.0, -1.0, 0.0, kv=8.0), 1.9)       # saturates at the max
    np.testing.assert_allclose(lift_command(5.0, 0.0, 0.0, kv=8.0), 0.0)         # cannot pull down
