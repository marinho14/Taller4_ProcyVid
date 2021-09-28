# Se importan las librerias necesarias
import cv2
import numpy as np
from Quadrilateral import Quadrilateral as ql  ## Se importa la clase Quadrilateral
from Corners import DetectCorners as DC        ## Se omporta la funcion para detectar esquinas


# Se define el main
if __name__ == '__main__':
   N=500         ## Se define el tamaño de la imagen cuadrada NxN
   imagen= ql(N)    ## Se crea una clase
   imagen= imagen.Generate()  ## Se usa el metodo Generate para generar una imagen con una figura de 4 lados
   draw, puntos=DC(imagen)         ## Se usa la función detectar esquinas y se iguala a una imagen
   print("Las posiciones de las esquinas (x,y) son: \n ", puntos) ## Se muestran las posiciones en x,y de las esquinas
   cv2.imshow("Image1", imagen)  ## Se muestra la imagen generada en Generate()
   cv2.imshow("Image", draw)     ## Se meusta la imagen con la detección de esquinas
   cv2.waitKey(0)


