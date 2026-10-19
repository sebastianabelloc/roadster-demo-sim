import numpy as np
from roadster.guidance import roll_reference, roll_quaternion, vertical_reference

G = 9.81


def test_roll_reference_before_during_after():
    # before the roll: level and still; after: full turn, still
    np.testing.assert_allclose(roll_reference(0.0, t_start=0.5, duration=2.0), (0.0, 0.0))
    np.testing.assert_allclose(roll_reference(3.0, t_start=0.5, duration=2.0), (-2 * np.pi, 0.0))


def test_roll_reference_halfway_is_upside_down_at_max_rate():
    phi, omega = roll_reference(1.5, t_start=0.5, duration=2.0)
    np.testing.assert_allclose(phi, -np.pi)
    np.testing.assert_allclose(omega, -np.pi**2 / 2.0)


def test_roll_reference_is_continuous_at_the_ends():
    # no jumps: just before/after the start and the end give the same values
    for t_edge in (0.5, 2.5):
        before = np.array(roll_reference(t_edge - 1e-6, t_start=0.5, duration=2.0))
        after = np.array(roll_reference(t_edge + 1e-6, t_start=0.5, duration=2.0))
        np.testing.assert_allclose(before, after, atol=1e-4)


def test_roll_quaternion_is_unit_and_matches_half_angle():
    for phi in np.linspace(-2 * np.pi, 0, 9):
        q = roll_quaternion(phi)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0)
    np.testing.assert_allclose(roll_quaternion(-np.pi), [0, -1, 0, 0], atol=1e-12)


def test_vertical_reference_descent_follows_braking_curve():
    # 4 m above the hover height, braking at 0.6 g: v = -sqrt(2 a d)
    v = vertical_reference(4.5, 2.6, None, z_hover=0.5, a_brake=0.6 * G)
    np.testing.assert_allclose(v, -np.sqrt(2 * 0.6 * G * 4.0))


def test_vertical_reference_never_nan_below_hover_height():
    assert vertical_reference(0.3, 3.0, None, z_hover=0.5) == 0.0


def test_vertical_reference_hover_then_land():
    assert vertical_reference(0.5, 4.0, 3.8, t_hover=0.8) == 0.0
    assert vertical_reference(0.5, 4.7, 3.8, t_hover=0.8, v_land=0.3) == -0.3
