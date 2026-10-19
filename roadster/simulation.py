import numpy as np
from scipy.integrate import solve_ivp
from roadster.dynamics import state_derivative 
from roadster.guidance import roll_reference, roll_quaternion
from roadster.control import pd_torque, select_thrusters
from roadster.thrusters import thruster_torque
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

def simulate_controlled(state0, inertia, thrust, kp, kd, deadband, lift=0.0, dt=0.02, t_max=10):
    """Simulates flight controller. Takes the initial state and calculates necessary inputs to reach the needed state.
       Then calculates the real position and repeats the same process every 20 milliseconds. Returns times, states and
       thrusters matrixes. Automatically stops at touchdown."""
    # converts inputs into numpy arrays
    state = np.asarray(state0, dtype=float)
    t = 0.0
    times, states, thrusters = [t],  [state.copy()], []
    # while loop from time to maximum simulation time
    while t < t_max:
        # gets the reference values (Guidance) from reference function
        phi_ref, omega_ref = roll_reference(t)
        q_ref = roll_quaternion(phi_ref)
        # extracts quaternion and omega values from state vector
        q = state[6:10]
        omega = state[10:13]
        # calculates the torque to be comanded based on current position and reference position. Uses the PD controller for this.
        torque_cmd = pd_torque(q, omega, q_ref, [omega_ref, 0, 0], kp, kd)
        # determines which thrusters to turn on and calculates resulting torque
        on = select_thrusters( torque_cmd, deadband)
        torque = thruster_torque(on, thrust)
        # calculates the new real position based on the torque exerted and interval time
        sol = solve_ivp(state_derivative, (t, t + dt), state, args=(inertia, torque, lift),
                        events=touchdown, max_step=0.005, rtol=1e-10, atol=1e-10)
        # extends vectors to save new data
        times.extend(sol.t[1:])
        states.extend(sol.y[:, 1:].T)
        thrusters.extend([on] * (len(sol.t)-1))
        # prepares for next loop
        state = sol.y[:, -1].copy()
        t = sol.t[-1]
        # normalizes quaternion to ensure the small residual error is not accumulated through the iterations
        state[6:10] = quaternion_normalize(state[6:10])
        # checks if the end requirements are met
        if sol.status == 1:
            break
    thrusters.insert(0, thrusters[0])
    return np.array(times), np.array(states).T, np.array(thrusters).T
        
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