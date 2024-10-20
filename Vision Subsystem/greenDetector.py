import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


def greenDetector():
    rawImg = cv2.imread("pictures\Divot1.jpg")
    plt.imshow(rawImg)
    plt.show()
    treshImg = gradient_thresh(rawImg)
    plt.imshow(treshImg)
    plt.show()


def gradient_thresh(img, thresh_min=25, thresh_max=100):
    """
    Apply sobel edge detection on input image in x, y direction
    """
    # 1. Convert the image to gray scale
    # 2. Gaussian blur the image
    # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
    # 4. Use cv2.addWeighted() to combine the results
    # 5. Convert each pixel to uint8, then apply threshold to get binary image

    # 1 - convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2 - Gaussian blur the image
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)  # 5 x 5 kernel size
    plt.imshow(blur)
    plt.show()

    # 3 - cv2.Sobel() to find derivatives
    ddepth = cv2.CV_16S
    gradient_x = cv2.Sobel(blur, ddepth, 1, 0)
    gradient_y = cv2.Sobel(blur, ddepth, 0, 1)

    # 4 - cv2.addWeighted() to combine the results
    combined = cv2.addWeighted(gradient_x, 0, gradient_y, 1, 0.0)

    # 5 - Convert each pixel to uint8 then apply threshold
    converted_uint8 = cv2.convertScaleAbs(combined)

    threshold = cv2.inRange(converted_uint8, thresh_min, thresh_max)
    threshold = threshold / 255

    ###

    return threshold


if __name__ == "__main__":
    greenDetector()
