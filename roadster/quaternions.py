import numpy as np

def quaternion_multiplication(p, q):
    """Multiplies two quaternions (w, x, y, z) p and q using Hamilton product.
       For unit quaternions it combines two rotations into one. Accepts any quaternions."""
    # convert input to numpy arrays
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # isolate quaternion components
    w1 = p[0]
    v1 = p[1:]
    w2 = q[0]
    v2 = q[1:]
    # use hamilton product
    # scalar part: w1*w2 - v1·v2
    w = w1 * w2 - np.dot(v1, v2)
    # vector part: w1*v2 + w2*v1 + v1×v2
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return np.concatenate(([w], v))

def quaternion_normalize(q):
    """Normalizes a quaternion and checks for division by 0 to raise error. Accepts any quaternion."""
    # convert input to numpy array
    q = np.asarray(q, dtype=float)
    # calculates norm of quaternion
    norm = np.sqrt(np.multiply(q, q).sum())
    # gives error if norm is 0 or very small
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero quaternion")
    # normalizes quaternion
    q_unit = q / norm
    return q_unit

def quaternion_derivative(q, omega):
    """Calculates the derivative of a quaternion given angular velocity vector omega.
       Accepts any quaternion and angular velocity vector on body frame in rad/s."""
    # convert input to numpy arrays
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    # create quaternion from angular velocity
    omega_quat = np.concatenate(([0], omega))
    # calculate derivative using Hamilton product
    q_dot = 0.5 * quaternion_multiplication(q, omega_quat)
    return q_dot

def quaternion_conjugate(q):
    """Returns the conjugate of a quaternion (w, x, y, z): same scalar part, vector part negated.
       For unit quaternions it is the inverse rotation."""
    # convert input to numpy array
    q = np.asarray(q, dtype=float)
    # keep scalar part and negate vector part
    return np.concatenate(([q[0]], -q[1:]))

def rotate_vector(q, v):
    """Rotates a vector v from the body frame to the inertial frame using the unit quaternion q (w, x, y, z).
       Uses v_inertial = q * (0, v) * q_conjugate. Returns only the vector part."""
    # convert input to numpy arrays
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    # create pure quaternion from the vector
    v_quat = np.concatenate(([0], v))
    # rotate using Hamilton product twice: q * v * q_conjugate
    v_rot = quaternion_multiplication(quaternion_multiplication(q, v_quat), quaternion_conjugate(q))
    # scalar part is 0, return the vector part
    return v_rot[1:]

if __name__ == "__main__":
    print("Example 1: Identity = ", quaternion_multiplication([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]))
    print("Example 2: Rotation i x j = ", quaternion_multiplication([0, 1, 0, 0], [0, 0, 1, 0]))
    print("Example 3: Rotation j x i = ", quaternion_multiplication([0, 0, 1, 0], [0, 1, 0, 0]))
    print("Example 4: Normalize [2, 0, 0, 0] = ", quaternion_normalize([2, 0, 0, 0]))
    print("Example 5: Derivative [1, 0, 0, 0] with omega [0, 0, 1] = ", quaternion_derivative([1, 0, 0, 0], [0, 0, 1]))