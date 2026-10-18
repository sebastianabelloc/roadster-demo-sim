import numpy as np
from scipy.integrate import solve_ivp
from roadster.dynamics import state_derivative 

def touchdown(t, state, inertia, torque):
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