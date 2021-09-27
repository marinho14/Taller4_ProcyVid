import cv2
import numpy as np
import scipy
import os
from scipy.stats import uniform as uni

# Uniforme entre 0 y 255



class Quadrilateral:  # Se crea la clase Basic Color

    def __init__(self, N):
        assert N % 2 == 0, "N debe ser par y entero"
        self.N= N

    def Generate(self):  # Se crea el metodo display properties
        image = np.zeros((self.N, self.N, 3), np.uint8)
        image[:, :, :] = [255, 255, 0]
        mean, var, skew, kurt = uni.stats(moments='mvsk', loc=0, scale=self.N / 2)
        pos_c1 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2))
        pos_c2 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2))
        pos_c2[0] = pos_c2[0] + self.N / 2
        pos_c3 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2))
        pos_c3[1] = pos_c3[1] + self.N / 2
        pos_c4 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2) + self.N / 2)

        magenta = (255, 0, 255)

        image = cv2.line(image, tuple(pos_c1), tuple(pos_c2), magenta, 5)
        image = cv2.line(image, tuple(pos_c1), tuple(pos_c3), magenta, 5)
        image = cv2.line(image, tuple(pos_c2), tuple(pos_c4), magenta, 5)
        image = cv2.line(image, tuple(pos_c3), tuple(pos_c4), magenta, 5)
        return image

