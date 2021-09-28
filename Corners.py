##Se definen las librerias necesarias
import sys
import numpy as np
from hough import Hough
from orientation_methods import gradient_map
import cv2


def DetectCorners(image):  ## Se define el metodo DetectCorners al cual le ingresa la imagen a analizar
    high_thresh = 600  ## Se define para el metodo Canny
    bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh,
                         L2gradient=True)  ## Se encuentra la imagen con sus bordes en blanco y negro

    hough = Hough(bw_edges)  ## Se usa la clase Hough  ingresando la imagen encontrada con Canny
    method = 1;
    if method == 1:
        accumulator = hough.standard_transform()
    elif method == 2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        theta, _ = gradient_map(image_gray)
        accumulator = hough.direct_transform(theta)
    else:
        sys.exit()

    ## Se definen parametros necesarios para encontrar los picos, los cuales se calcularon mediante sintonización
    acc_thresh = 60
    N_peaks = 10  ## Se define un maximo de 10 esquinas por encontrar por imagen
    nhood = [50, 50]
    peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)  ## Se usa el metodo Find_peaks

    _, cols = image.shape[:2]
    image_draw = np.copy(image)  ## Se define una copia de la iamgen original
    pendiente = []  ## Se crean listas vacias
    corte = []
    puntos = np.empty((2, 2), dtype=int)
    i = 0
    for peak in peaks:  ## Se recorre peaks, este for se hara de acuerdo a la cantidad de lineas que encuentre
        rho = peak[0]  ## Se define rho
        theta_ = hough.theta[peak[1]]  ## Se define theta

        theta_pi = np.pi * theta_ / 180  ## Se define el theta en rad
        theta_ = theta_ - 180
        a = np.cos(theta_pi)  ## coseno de theta
        b = np.sin(theta_pi)  ## Seno de theta
        xd = a * rho + hough.center_x  ## Se define el x
        yd = b * rho + hough.center_y  ## Se define el y
        c = -rho

        pendiente.append(-1 / np.tan(
            theta_pi))  ## Se agrega la pendiente a la lista, el número de pendientes sera igual al de lineas encontradas
        corte.append(yd - pendiente[
            i] * xd)  ## Se agrega el corte a la lista, el número de cortes sera igual al de lineas encontradas
        i = i + 1

        # Se dibijan las lineas sobre las rectas encontradas

        x1 = int(round(xd + cols * (-b)))
        y1 = int(round(yd + cols * a))
        x2 = int(round(xd - cols * (-b)))
        y2 = int(round(yd - cols * a))

        if np.abs(theta_) < 80:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)
        elif np.abs(theta_) > 100:
            image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)
        else:
            if theta_ > 0:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)
            else:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 0], thickness=2)

    for i in range(len(corte)):  ## Se recorre la lista de cortes y pendientes
        for j in range(i, len(corte)):
            ## Se definen parametros de la resta de los cortes y pendientes
            c = (corte[i] - corte[j])
            p = (pendiente[j] - pendiente[i])
            ## Se encuentran los puntos donde esta el poligono
            puntos_poligono = np.where(bw_edges >= 1)
            puntos_poligono = np.array([puntos_poligono[1].tolist(), puntos_poligono[0].tolist()])
            if p != 0:  ## Solo se entra si la resta de pendientes es diferente de 0
                X = c / p  ## Se encuentra la posición en X donde hay intersección entre las lineas
                Y = pendiente[i] * X + corte[
                    i]  ## Se encuentra la posición en Y donde hay intersección entre las lineas
                punto = np.array([X, Y]).reshape(2, 1)  ## Se define el punto de la intersección
                ## Se encuentra la distancia mas pequeña del poligono al punto, esto se hace ya que pueden haber intersecciones
                ## lejanas al poligono ya que las lineas se expanden por toda la imagen
                dist = np.sqrt(((puntos_poligono - punto) ** 2).sum(axis=0)).min()
                if dist < 10:  ## Solo se entra si la distancia es menor a un valor, esta se encontro por sintonizacion
                    puntos = np.append(puntos, [[int(X), int(Y)]], axis=0)
                    image_draw = cv2.circle(image_draw, (int(X), int(Y)), 10, (45, 0, 0),
                                            2)  ## Se grafica un circulo en donde se encontro la intersección
                    # print(X,Y)
    puntos = np.delete(puntos, 0, axis=0)
    puntos = np.delete(puntos, 0, axis=0)
    return image_draw, puntos  ## Se retorna la imagen con las lineas y las esquinas encerradas y sus posiciones
