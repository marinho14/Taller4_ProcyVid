import cv2     ## Se llaman las librerias necesarias
import numpy as np
from scipy.stats import uniform as uni


class Quadrilateral:  # Se crea la clase Quadrilateral

    def __init__(self, N):   ## Se define el constructor
        assert N % 2 == 0, "N debe ser par y entero"   ## Se asegura que N sea un numero par de lo contrario se arrojara un error
        self.N = N

    def Generate(self):  # Se crea el metodo Generate
        image = np.zeros((self.N, self.N, 3), np.uint8)  ## Se crea un Np array de NxNx3
        image[:, :, :] = [255, 255, 0]  ## Se define el color cyan
        mean, var, skew, kurt = uni.stats(moments='mvsk', loc=0, scale=self.N / 2)  ## Se usa la libreria Scipy para generar los puntos en cada cuadrante aleatoriamente
        pos_c1 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2)) ## Se define el punto del primer cuadrante
        pos_c2 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2))
        pos_c2[0] = pos_c2[0] + self.N / 2                         ## Se define el punto del segundo cuadrante
        pos_c3 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2))
        pos_c3[1] = pos_c3[1] + self.N / 2                         ## Se define el punto del tercer cuadrante
        pos_c4 = np.int_(uni.rvs(loc=0, scale=self.N / 2, size=2) + self.N / 2) ## Se define el punto del cuarto cuadrante

        magenta = (255, 0, 255)   ## Se define el color magenta

        image = cv2.line(image, tuple(pos_c1), tuple(pos_c2), magenta, 5)  ## Se unen los puntos para generar la figura
        image = cv2.line(image, tuple(pos_c1), tuple(pos_c3), magenta, 5)
        image = cv2.line(image, tuple(pos_c2), tuple(pos_c4), magenta, 5)
        image = cv2.line(image, tuple(pos_c3), tuple(pos_c4), magenta, 5)
        return image
