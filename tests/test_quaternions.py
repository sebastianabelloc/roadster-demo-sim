import numpy as np
import pytest
from roadster.quaternions import quaternion_conjugate, quaternion_multiplication, quaternion_normalize, quaternion_derivative, rotate_vector

def test_quaternion_identity_multiplication():
    q1 = np.array([1, 0, 0, 0])  # Identity quaternion
    q2 = np.array([0.5, 0.5, 0.5, 0.5])  # Arbitrary quaternion
    result = quaternion_multiplication(q1, q2)
    np.testing.assert_allclose(result, [0.5, 0.5, 0.5, 0.5])

def test_quaternion_rotation_i_times_j():
    q1 = np.array([0, 1, 0, 0])  # i
    q2 = np.array([0, 0, 1, 0])  # j
    result = quaternion_multiplication(q1, q2)
    np.testing.assert_allclose(result, [0, 0, 0, 1])  # i x j = k

def test_quaternion_rotation_j_times_i():
    q1 = np.array([0, 0, 1, 0])  # j
    q2 = np.array([0, 1, 0, 0])  # i
    result = quaternion_multiplication(q1, q2)
    np.testing.assert_allclose(result, [0, 0, 0, -1])  # j x i = -k

def test_quaternion_normalization():
    q = np.array([5, 0, 0, 0])
    result = quaternion_normalize(q)
    np.testing.assert_allclose(result, [1, 0, 0, 0])

def test_normalize_zero_quaternion_raises():
    with pytest.raises(ValueError):
        quaternion_normalize([0, 0, 0, 0])

def test_quaternion_derivative():
    q = np.array([1, 0, 0, 0])  # Identity quaternion
    omega = np.array([0, 0, 1])  # Angular velocity around z-axis
    result = quaternion_derivative(q, omega)
    np.testing.assert_allclose(result, [0, 0, 0, 0.5])  # Expected derivative

def test_quaternion_conjugate():
    result = quaternion_conjugate([1, 2, 3, 4])
    np.testing.assert_allclose(result, [1, -2, -3, -4])

def test_conjugate_undoes_rotation():
    # q ⊗ q* must give the identity (no rotation)
    q = [0.70710678, -0.70710678, 0, 0]
    result = quaternion_multiplication(q, quaternion_conjugate(q))
    np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-8)

def test_rotate_vector_identity():
    # identity quaternion: the vector does not change
    result = rotate_vector([1, 0, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(result, [1, 2, 3])

def test_rotate_vector_roll_90():
    # roll +90 deg about x takes the body z axis to -y
    c = np.cos(np.pi / 4)
    result = rotate_vector([c, c, 0, 0], [0, 0, 1])
    np.testing.assert_allclose(result, [0, -1, 0], atol=1e-12)

