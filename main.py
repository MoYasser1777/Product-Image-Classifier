import cv2
import numpy as np
import os

from Classifier import *

train()

image = cv2.imread("./test3.jpg", cv2.IMREAD_GRAYSCALE)
print(get_prediction(image))