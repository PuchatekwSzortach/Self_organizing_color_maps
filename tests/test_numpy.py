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


def test_euclidean_distance_broadcasting_for_grid_computations():
    """
    Checking a 3D point is correctly broadcasted for distance computations when used with
    3D matrix
    """

    point = np.array([1, 0])

    print()
    print("point shape")
    print(point.shape)

    matrix = np.array([
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
        [[2, 0], [2, 1], [2, 2]],
    ])

    print("matrix shape")
    print(matrix.shape)

    expected = np.array([
        [1, np.sqrt(2), np.sqrt(5)],
        [0, 1, 2],
        [1, np.sqrt(2), np.sqrt(5)]
    ])

    actual = np.linalg.norm(point - matrix, axis=2)

    assert np.all(expected == actual)


def test_subtraction_broadcasting():
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


def test_3d_matrix_and_2d_matrix_multiplication():
    """
    Check that we can use a 2D matrix a to multiply a 2D matrix of points b (so 3D matrix),
    so that points in b are scaled by values in corresponding elements of a
    """

    a = np.array([[1, 2, 3],
                  [3, 2, 1]])

    b = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                  [[4, 4, 4], [5, 5, 5], [10, 10, 10]]])

    expected = np.array([[[1, 1, 1], [4, 4, 4], [9, 9, 9]],
                         [[12, 12, 12], [10, 10, 10], [10, 10, 10]]])

    actual = a[..., np.newaxis] * b
    assert np.all(expected == actual)


def test_mesh_grid():

    xy = np.rollaxis(np.indices([3, 3]), 0, 3)

    expected = np.array([
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
        [[2, 0], [2, 1], [2, 2]],
    ])

    assert np.all(expected == xy)


def test_broadcasting_2d_matrix_to_3d():

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
