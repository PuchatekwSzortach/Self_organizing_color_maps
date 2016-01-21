import numpy as np
import itertools
import cv2


class SOCMTrainer:
    """
    A class for training a Self Organizing Color Map
    """

    def __init__(self, shape, iterations_number):
        """
        :param shape: a 2 elements list that specifies map size
        :param iterations_number: number of iterations map will be trained on iput
        """
        if len(shape) != 2:
            raise ValueError("Shape must have 2 dimensions")

        # Each color has 3 elements
        map_shape = shape + [3]

        self.map = np.random.random(map_shape)
        self.iterations_number = iterations_number

        self.coordinates_grid = self.get_coordinates_grid()

    def fit(self, training_data):

        for iteration_index in range(self.iterations_number):

            print("Starting iteration {}".format(iteration_index))

            for element in training_data:
                self.fit_element(element, iteration_index)

            cv2.imshow("map", cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(self.map))))
            cv2.waitKey(30)

    def fit_element(self, element, iteration_number):

        # Compute distances to all elements of map
        distance_map = np.linalg.norm(element - self.map, axis=2)

        best_matching_unit_indices = np.unravel_index(np.argmin(distance_map), distance_map.shape)

        neighborhood_function = self.get_neighborhood_function(best_matching_unit_indices, iteration_number)
        neighborhood_function_replicated = np.dstack((neighborhood_function, neighborhood_function, neighborhood_function))

        learning_rate = np.exp(-iteration_number / self.iterations_number)
        # learning_rate = (self.iterations_number - iteration_number) / self.iterations_number

        weight_increment = neighborhood_function_replicated * learning_rate * (element - self.map)

        self.map += weight_increment

    def get_neighborhood_function(self, best_matching_unit_indices, iteration_number):

        # gaussian_width = self.map.shape[0]
        gaussian_width = iteration_number / self.iterations_number
        distance_to_best_match = np.linalg.norm(best_matching_unit_indices - self.coordinates_grid, axis=2)

        exponent = -1 * (gaussian_width * distance_to_best_match)**2
        neighborhood_function = np.exp(exponent)

        return neighborhood_function

    def get_coordinates_grid(self):

        y = range(0, self.map.shape[0])
        x = range(0, self.map.shape[1])

        yx_list = list(itertools.product(y, x))
        yx_vector = np.array(yx_list)

        yx_matrix = np.array(yx_vector).reshape((self.map.shape[0], self.map.shape[1], 2))
        return yx_matrix
