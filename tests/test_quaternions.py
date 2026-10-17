import numpy as np
import pytest
from roadster.quaternions import quaternion_multiplication, quaternion_normalize, quaternion_derivative

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