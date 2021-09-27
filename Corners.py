import os
import sys
from enum import Enum
import numpy as np
from hough import Hough
from orientation_methods import gradient_map
import cv2
import math


def DetectCorners(image, method):
    high_thresh = 600
    bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)

    hough = Hough(bw_edges)
    if method == 1:
        accumulator = hough.standard_transform()
    elif method == 2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        theta, _ = gradient_map(image_gray)
        accumulator = hough.direct_transform(theta)
    else:
        sys.exit()

    acc_thresh = 60
    N_peaks = 10
    nhood = [50 , 50]
    peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

    _, cols = image.shape[:2]
    image_draw = np.copy(image)
    pendiente = []
    corte = []
    i = 0
    for peak in peaks:
        rho = peak[0]
        theta_ = hough.theta[peak[1]]

        theta_pi = np.pi * theta_ / 180
        theta_ = theta_ - 180
        a = np.cos(theta_pi)
        b = np.sin(theta_pi)
        xd = a * rho + hough.center_x
        yd = b * rho + hough.center_y
        c = -rho

        pendiente.append(-1 / np.tan(theta_pi))
        corte.append(yd - pendiente[i] * xd)
        i = i + 1

        x1 = int(round(xd + cols * (-b)))
        y1 = int(round(yd + cols * a))
        x2 = int(round(xd - cols * (-b)))
        y2 = int(round(yd - cols * a))

        # image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)

        if np.abs(theta_) < 80:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)
        elif np.abs(theta_) > 100:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)
        else:
            if theta_ > 0:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)
            else:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)

    for i in range(len(corte)):
        for j in range(i, len(corte)):
            c = (corte[i] - corte[j])
            p = (pendiente[j] - pendiente[i])
            puntos_poligono = np.where(bw_edges >= 1)
            puntos_poligono = np.array([puntos_poligono[1].tolist(), puntos_poligono[0].tolist()])
            if c != 0 and p != 0:
                X = ((corte[i] - corte[j]) / (pendiente[j] - pendiente[i]))
                Y = pendiente[i] * X + corte[i]
                punto = np.array([X, Y]).reshape(2, 1)
                dist = np.sqrt(((puntos_poligono - punto) ** 2).sum(axis=0)).min()
                if dist < 10:
                    image_draw = cv2.circle(image_draw, (int(X), int(Y)), 10, (45, 0, 0), 2)

    return image_draw
