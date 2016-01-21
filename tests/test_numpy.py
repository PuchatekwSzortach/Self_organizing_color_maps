"""
Various numpy tests, mainly to make sure broadcasting 1D and 2D data to 3D matrices behaves as expected,
since a lot of operations we de in algorithms module rely on broadcasting and vectorized operations
"""

import numpy as np


def test_euclidean_distance_broadcasting():
    """
    Checking a 3D point is correctly broadcasted for distance computations when used with
    3D matrix
    """

    point = np.array([1, 1, 1])

    matrix = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
    ])

    expected = np.array([
        [0, np.sqrt(3), np.sqrt(12), np.sqrt(27)],
        [np.sqrt(27), np.sqrt(48), np.sqrt(75), np.sqrt(108)]
    ])

    actual = np.linalg.norm(point - matrix, axis=2)

    assert np.all(expected == actual)


def test_subtraction_broadcasting():

    point = np.array([1, 1, 1])

    matrix = np.array([
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
    ])

    expected = np.array([
        [[0, 0, 0], [-1, -1, -1], [-2, -2, -2], [-3, -3, -3]],
        [[-3, -3, -3], [-4, -4, -4], [-5, -5, -5], [-6, -6, -6]],
    ])

    actual = point - matrix

    assert np.all(expected == actual)


def test_2d_matrices_element_wise_multiplication():

    a = np.array([[1, 2, 3],
                  [4, 5, 6]])

    b = np.array([[3, 2, 1],
                  [4, 5, 10]])

    expected = np.array([[3, 4, 3], [16, 25, 60]])
    actual = a * b

    assert np.all(expected == actual)


def test_stacking_2d_matrix_depth_wise():

    input = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ])

    expected = np.array(
        [
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
            [[6, 6, 6], [7, 7, 7], [8, 8, 8]],
        ]
    )

    actual = np.dstack((input, input, input))

    assert np.all(expected == actual)
