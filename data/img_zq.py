import cv2 as cv
import numpy as np

img = cv.imread(r"s", 0)

#131 79   3
# img = img[]

cv.imshow("img", img)

cv.waitKey()
