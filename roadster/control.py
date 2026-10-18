import numpy as np
from roadster.quaternions import quaternion_conjugate, quaternion_multiplication

def attitude_error(q, q_ref):
    """Calculates the attitude error quaternion between the current attitude quaternion q and the reference quaternion q_ref. 
    Checks for the sign of the scalar part to ensure the shortest rotation."""
    # convert input to numpy arrays
    q = np.asarray(q, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    # calculate attitude error using Hamilton product
    q_err = quaternion_multiplication(quaternion_conjugate(q_ref), q)
    # ensure scalar part is non-negative
    if q_err[0] < 0:
        q_err = -q_err  
    return q_err

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("No error:      ", attitude_error([1, 0, 0, 0], [1, 0, 0, 0]), "expected [1, 0, 0, 0]")
    print("q vs -q:       ", attitude_error([1, 0, 0, 0], [-1, 0, 0, 0]), "expected [1, 0, 0, 0]")
    print("-90 deg on x:  ", attitude_error([0.707, -0.707, 0, 0], [1, 0, 0, 0]), "expected [0.707, -0.707, 0, 0]")