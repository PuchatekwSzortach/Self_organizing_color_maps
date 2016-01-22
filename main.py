import numpy as np
import cv2

import socm.algorithms

if __name__ == "__main__":

    socm_trainer = socm.algorithms.SOCMTrainer([200, 200], 100)
    training_samples = np.random.random([100, 3])
    socm_trainer.train(training_samples)

    map = cv2.pyrUp(socm_trainer.map)

    cv2.imshow("map", map)
    cv2.waitKey(0)

