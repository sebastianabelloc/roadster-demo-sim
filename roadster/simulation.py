import numpy as np
from scipy.integrate import solve_ivp
from roadster.dynamics import state_derivative 
from roadster.guidance import roll_reference, roll_acceleration, roll_quaternion, vertical_reference, lateral_correction
from roadster.control import pd_torque, allocate_thrusters, lift_command
from roadster.thrusters import thruster_torque, thruster_force
from roadster.quaternions import quaternion_normalize

def touchdown(t, state, *args):
    """Determines when the Roadster has landed to stop the simulation. The Roadster is considered to have landed when its z position reaches altitude 0 when descending."""
    # convert input to numpy array
    state = np.asarray(state, dtype=float)
    # isolate position from state vector
    r = state[0:3]
    # return the z position (height) of the Roadster
    return r[2]
    
touchdown.terminal = True
touchdown.direction = -1  # only trigger when z is decreasing

def simulate_roadster(state0, inertia, torque, t_max=10, max_step=0.01, rtol=1e-6, atol=1e-9):
    """Simulates the Roadster's motion using solve_ivp. Needs initial state vector, inertia matrix, torque vector and maximum simulation time. Maximum time is 10 seconds 
    by default, however it is optional as the simualtion will automatically stop when the Roadster lands."""
    # convert input to numpy arrays
    state0 = np.asarray(state0, dtype=float)
    inertia = np.asarray(inertia, dtype=float)
    torque = np.asarray(torque, dtype=float)
    # define time span for simulation
    t_span = (0, t_max)
    # solve the initial value problem using RK45 method
    sol = solve_ivp(state_derivative, t_span, state0, args=(inertia, torque), events=touchdown, max_step=max_step, rtol=rtol, atol=atol)
    return sol

def simulate_controlled(state0, inertia, thrust, kp, kd, deadband, lift_flight=0.0,
                        kv=8.0, t_brake=2.35, z_hover=0.5, a_brake=0.6 * 9.81, dt=0.02, t_max=10,
                        deadband_hold=1500, t_roll_end=2.35, side_thrust=3000.0, mass=2000.0,
                        feedforward=True, lateral=True, k_y=0.06, k_vy=0.20,
                        t_roll_start=0.85, roll_duration=1.5, v_launch=None, lift_launch=1.9):
    """Simulates flight controller. Takes the initial state and calculates necessary inputs to reach the needed state.
       Then calculates the real position and repeats the same process every 20 milliseconds. Returns times, states and
       thrusters matrixes. Automatically stops at touchdown.
       thrust is the maximum thrust of each of the 4 bottom nozzles (N) and side_thrust the one of the 2 side nozzles (N).
       The thrusters matrix holds the throttle (0 to 1) of the 6 nozzles and lifts the bottom thrust as a fraction of the weight.
       With v_launch the car starts on the ground (vz = 0) and the bottom nozzles push it up with lift_launch until it
       climbs at v_launch; then they cut and it flies freely until braking."""
    # converts inputs into numpy arrays
    state = np.asarray(state0, dtype=float)
    t = 0.0
    times, states, thrusters = [t],  [state.copy()], []
    t_hover_start = None     # set when the car first reaches z_hover
    lifts = []               # lift used in each sample, for plots and the animation
    launching = v_launch is not None    # powered take-off phase
    # while loop from time to maximum simulation time
    while t < t_max:
        # gets the reference values (Guidance) from reference function
        phi_ref, omega_ref = roll_reference(t, t_roll_start, roll_duration)
        # after the roll: lateral position loop, a small extra roll so the car comes back to the centre line
        if lateral and t >= t_roll_end:
            phi_ref = phi_ref + lateral_correction(state[1], state[4], k_y, k_vy)
        q_ref = roll_quaternion(phi_ref)
        # extracts quaternion and omega values from state vector
        q = state[6:10]
        omega = state[10:13]
        # calculates the torque to be comanded based on current position and reference position. Uses the PD controller for this.
        torque_cmd = pd_torque(q, omega, q_ref, [omega_ref, 0, 0], kp, kd)
        # feedforward: the torque the roll profile needs (I * alpha), so the roll ends exactly level when braking starts
        if feedforward:
            torque_cmd = torque_cmd + inertia @ np.array([roll_acceleration(t, t_roll_start, roll_duration), 0.0, 0.0])
        # vertical guidance and control
        # launch: full push from the bottom nozzles until the car climbs at v_launch, then engine cut
        if launching and state[5] >= v_launch:
            launching = False
        if launching:
            lift = lift_launch
        elif t < t_brake:
            lift = lift_flight
        else:
            z, vz = state[2], state[5]
            if t_hover_start is None and z <= z_hover + 0.05:
                t_hover_start = t
            vz_ref = vertical_reference(z, t, t_hover_start, z_hover=z_hover, a_brake=a_brake)
            a_ff = a_brake if t_hover_start is None else 0.0
            lift = lift_command(vz, vz_ref, a_ff, kv)
        # the bottom nozzles push along the body z axis: divide by its vertical component to get the vertical lift
        body_z_up = 1 - 2 * (q[1]**2 + q[2]**2)
        force_z = lift * mass * 9.81 / max(body_z_up, 0.5)
        # control allocation: throttle of the 4 bottom nozzles and on/off of the 2 side ones
        # attitude hold after the roll: wider yaw deadband so tiny errors do not make the side nozzles chatter
        active_deadband = deadband if t < t_roll_end else deadband_hold
        max_thrust = np.array([thrust] * 4 + [side_thrust] * 2)
        on = allocate_thrusters(force_z, torque_cmd, active_deadband, thrust)
        torque = thruster_torque(on, max_thrust)
        force = thruster_force(on, max_thrust)
        # calculates the new real position based on the torque exerted and interval time
        sol = solve_ivp(state_derivative, (t, t + dt), state, args=(inertia, torque, 0.0, force, mass),
                        events=touchdown, max_step=0.005, rtol=1e-10, atol=1e-10)
        # extends vectors to save new data
        times.extend(sol.t[1:])
        states.extend(sol.y[:, 1:].T)
        thrusters.extend([on] * (len(sol.t)-1))
        lifts.extend([force[2] / (mass * 9.81)] * (len(sol.t) - 1))
        # prepares for next loop
        state = sol.y[:, -1].copy()
        t = sol.t[-1]
        # normalizes quaternion to ensure the small residual error is not accumulated through the iterations
        state[6:10] = quaternion_normalize(state[6:10])
        # checks if the end requirements are met
        if sol.status == 1:
            break
    thrusters.insert(0, thrusters[0])
    lifts.insert(0, lifts[0])
    return np.array(times), np.array(states).T, np.array(thrusters).T, np.array(lifts)

def add_ground_roll(times, states, thrusters, lifts, t_before=1.5, t_after=1.5, dt=0.02):
    """Adds kinematic ground segments: the car driving forward before take-off and rolling on after touchdown exclusively on positive x direction.
       No dynamics on the ground: constant speed, level attitude, thrusters off. Same output format as simulate_controlled."""
    vx = states[3, 0]                        # speed when it leaves the ground
    # --- before take-off: times from -t_before to 0 (0 excluded, the flight already has it)
    t_pre = np.arange(-t_before, 0.0, dt)
    pre = np.zeros((13, len(t_pre)))
    pre[0] = vx * t_pre                      # x goes from negative values up to 0
    pre[3] = vx                              # moving forward
    pre[6] = 1.0                             # level: quaternion (1, 0, 0, 0)
    # --- after touchdown: keep rolling at the landing horizontal speed
    t_end, x_end, vx_end = times[-1], states[0, -1], states[3, -1]
    t_post = t_end + np.arange(dt, t_after + dt, dt)
    post = np.zeros((13, len(t_post)))
    post[0] = x_end + vx_end * (t_post - t_end)
    post[1] = states[1, -1]                  # keep the small lateral offset, if any
    post[3] = vx_end
    post[6] = np.sign(states[6, -1])         # level, same sign as the last flight quaternion (-1 after the roll)
    # --- join the three segments
    times_all = np.concatenate([t_pre, times, t_post])
    states_all = np.concatenate([pre, states, post], axis=1)
    thrusters_all = np.concatenate([np.zeros((thrusters.shape[0], len(t_pre))), thrusters,
                                    np.zeros((thrusters.shape[0], len(t_post)))], axis=1)
    lifts_all = np.concatenate([np.zeros(len(t_pre)), lifts, np.zeros(len(t_post))])
    return times_all, states_all, thrusters_all, lifts_all


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    # Box approximation of the Roadster: m = 2000 kg, L = 4.2 m, B = 2.0 m, H = 1.2 m
    m, L, B, H = 2000.0, 4.2, 2.0, 1.2
    I = np.diag([m / 12 * (B**2 + H**2),
                 m / 12 * (L**2 + H**2),
                 m / 12 * (L**2 + B**2)])
    state0 = [0, 0, 0,   8, 0, 12.5,   1, 0, 0, 0,   0, 0, 0]
    sol = simulate_roadster(state0, I, [0, 0, 0], t_max=10, max_step=0.01, rtol=1e-6, atol=1e-9)
    print("Tesla Roadster landed at: ", sol.t[-1], "seconds with final state: ", sol.y[:, -1])
    # closed loop demo
    state0_ctrl = [0, 0, 0,   6.3, 0, 0.0,   1, 0, 0, 0,   0.1, 0.25, 0.05]
    times, states, thrusters, lifts = simulate_controlled(state0_ctrl, I, thrust=10000, kp=5e4, kd=3e3,
                                                   deadband=400, lift_flight=0.0, v_launch=7.0)
    print("Flight time:      ", times[-1], "s   expected ~4.9")
    print("Final x:          ", states[0, -1], "m   expected ~35")
    print("Final quaternion: ", states[6:10, -1], "  expected ~(-1, 0, 0, 0)")
    print("Final omega:      ", states[10:13, -1], "rad/s   expected ~0")
    print("Touchdown vz:     ", states[5, -1], "m/s   expected ~-0.6")
    print("Max lift:         ", lifts.max(), "   expected <= 2.04 (4 bottom nozzles at full thrust)")
    # add the ground segments before take-off and after touchdown
    t_all, s_all, th_all, lf_all = add_ground_roll(times, states, thrusters, lifts)
    print("Full run from", t_all[0], "to", t_all[-1], "s,", len(t_all), "samples   expected -1.5 to ~6.4 s")
    print("x from", s_all[0, 0], "to", s_all[0, -1], "m   expected -9.45 to ~45")
