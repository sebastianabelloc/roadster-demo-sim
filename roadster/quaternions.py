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

if __name__ == "__main__":

    print("Example 1: Identity", quaternion_multiplication([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]))
    print("Example 2: Rotation i x j", quaternion_multiplication([0, 1, 0, 0], [0, 0, 1, 0]))
    print("Example 3: Rotation j x i", quaternion_multiplication([0, 0, 1, 0], [0, 1, 0, 0]))

