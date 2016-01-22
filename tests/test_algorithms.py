import numpy as np
import socm.algorithms


def test_get_2d_coordinates_grid():

    shape = [2, 3]

    expected = np.array([
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
    ])

    actual = socm.algorithms.get_2d_coordinates_grid(shape)

    assert np.all(expected == actual)
