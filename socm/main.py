import numpy as np
import cv2

import matplotlib.pyplot as plt

import algorithms

if __name__ == "__main__":

    socm_trainer = algorithms.SOCMTrainer([50, 50], 200)
    training_samples = np.random.random([100, 3])
    socm_trainer.fit(training_samples)

    map = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(socm_trainer.map)))

    cv2.imshow("map", map)
    cv2.waitKey(0)
