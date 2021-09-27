# This is a sample Python script.
import cv2
# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from Quadrilateral import Quadrilateral as ql
from Corners import DetectCorners as DC


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   imagen= ql(500)
   imagen= imagen.Generate()
   draw=DC(imagen, 1)
   cv2.imshow("Image1", imagen)
   cv2.imshow("Image", draw)
   cv2.waitKey(0)


