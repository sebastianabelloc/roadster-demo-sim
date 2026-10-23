import numpy as np
import pytest
from roadster.simulation import simulate_roadster, simulate_controlled, add_ground_roll

M, L, B, H = 2000.0, 4.2, 2.0, 1.2
INERTIA = np.diag([M / 12 * (B**2 + H**2),
                   M / 12 * (L**2 + H**2),
                   M / 12 * (L**2 + B**2)])
DEMO_STATE0 = [0, 0, 0,   6.3, 0, 0.0,   1, 0, 0, 0,   0.1, 0.25, 0.05]
DEMO = dict(thrust=10000, kp=5e4, kd=3e3, deadband=400, lift_flight=0.0, v_launch=7.0)


@pytest.fixture(scope="module")
def demo_run():
    # the full demo is slow-ish (~150 control cycles), so run it once for all tests
    return simulate_controlled(DEMO_STATE0, INERTIA, **DEMO)


def test_no_control_matches_open_loop():
    # kp = kd = 0 and no braking: the loop must reproduce the plain free-flight simulation
    state0 = [0, 0, 0,   8, 0, 12.5,   1, 0, 0, 0,   0.3, 0.8, 0.1]
    times, states, _, _ = simulate_controlled(state0, INERTIA, thrust=10000, kp=0, kd=0,
                                              deadband=400, lift_flight=0.0, t_brake=100,
                                              feedforward=False, lateral=False)
    sol = simulate_roadster(state0, INERTIA, [0, 0, 0])
    np.testing.assert_allclose(times[-1], sol.t[-1], rtol=1e-6)
    np.testing.assert_allclose(states[:, -1], sol.y[:, -1], atol=1e-4)


def test_outputs_have_matching_lengths(demo_run):
    times, states, thrusters, lifts = demo_run
    n = len(times)
    assert states.shape == (13, n)
    assert thrusters.shape == (6, n)
    assert lifts.shape == (n,)
    assert np.all(np.diff(times) > 0)          # no repeated instants


def test_demo_completes_the_roll_and_lands_level(demo_run):
    _, states, _, _ = demo_run
    q_final = states[6:10, -1]
    w, x, y, z = q_final
    tilt_deg = np.degrees(np.arccos(1 - 2 * (x**2 + y**2)))   # angle between body z and vertical (yaw not included)
    assert q_final[0] < 0                      # full turn: quaternion ends near -1
    assert tilt_deg < 3.0                      # the hold deadband accepts a couple of degrees to avoid chattering


def test_attitude_hold_does_not_chatter(demo_run):
    # the side (yaw) nozzles are on/off: after the roll only a few short touches
    times, _, thrusters, _ = demo_run
    after_roll = times >= 2.35
    pulses = 0
    for i in (4, 5):
        on = thrusters[i, after_roll] > 0
        pulses += int(np.sum(on[1:] & ~on[:-1]) + on[0])
    assert pulses <= 10


def test_throttles_between_0_and_1(demo_run):
    # every nozzle only pushes and never above its maximum
    _, _, thrusters, _ = demo_run
    assert thrusters.min() >= 0.0 and thrusters.max() <= 1.0 + 1e-12


def test_demo_soft_landing_within_thrust_limit(demo_run):
    _, states, _, lifts = demo_run
    np.testing.assert_allclose(states[5, -1], -0.6, atol=0.05)   # touchdown vertical speed
    assert lifts.max() <= 4 * 10000 / (M * 9.81) + 1e-9      # 4 bottom nozzles at full thrust
    assert lifts.min() >= 0.0


def test_demo_quaternion_stays_unit(demo_run):
    _, states, _, _ = demo_run
    np.testing.assert_allclose(np.linalg.norm(states[6:10], axis=0), 1.0, atol=1e-9)


def test_ground_roll_segments(demo_run):
    times, states, thrusters, lifts = demo_run
    t_all, s_all, th_all, lf_all = add_ground_roll(times, states, thrusters, lifts, t_before=1.5, t_after=1.5)
    assert np.all(np.diff(t_all) > 0)                        # time keeps increasing, no duplicates
    np.testing.assert_allclose(t_all[0], -1.5)
    np.testing.assert_allclose(s_all[0, 0], -1.5 * 6.3)      # 1.5 s before take-off, 9.45 m behind the take-off point
    assert np.all(s_all[2, t_all < 0] == 0)                  # on the ground before take-off
    assert np.all(s_all[2, t_all > times[-1]] == 0)          # and after touchdown
    assert not th_all[:, t_all < 0].any()                    # thrusters off on the ground
    assert s_all.shape[1] == th_all.shape[1] == len(lf_all) == len(t_all)


def test_roll_ends_level_before_braking(demo_run):
    # with the feedforward torque the car is level when the braking starts (no tilted braking push)
    times, states, _, _ = demo_run
    k = np.searchsorted(times, 2.35)
    w, x, y, z = states[6:10, k]
    roll_deg = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2)))
    assert abs(roll_deg) < 2.0


def test_take_off_is_powered(demo_run):
    # the car leaves the ground with vz = 0: only the bottom nozzles can lift it (no initial kick)
    times, states, thrusters, lifts = demo_run
    assert states[5, 0] == 0.0
    assert lifts[0] > 1.0                                    # more than the weight at take-off
    climbing = np.cumsum(states[5] >= DEMO["v_launch"]) == 0   # from take-off until the engine cut
    assert np.all(thrusters[:4, climbing] > 0.5)             # the 4 bottom nozzles are firing
    assert states[2].max() < 11.0                            # apex around 10 m, like the demo


def test_car_stays_on_the_runway(demo_run):
    # the red lines are 4.3 m from the centre line: the car must stay well inside
    _, states, _, _ = demo_run
    assert np.abs(states[1]).max() < 1.5

