import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import data
from skimage import filters
from skimage.color import rgb2gray



def greenDetector():
    rawImg = cv2.imread("pictures\Divot1.jpg")
    # divotDetectorGrayBin()

    plt.imshow(rawImg)
    plt.show()
    # cv2.imshow("Raw Image", rawImg)
    # cv2.waitKey(0)
    treshImg = gradient_thresh(rawImg)
    plt.imshow(treshImg)
    plt.show()
    # cv2.imshow("Tresh Image", treshImg)
    # cv2.waitKey(0)

    start_time = time.perf_counter()
    maskedImg = color_thresh(rawImg)
    end_time = time.perf_counter()
    # with np.set_printoptions(threshold=np.inf):
    #     print(maskedImg[2000])
    print("--- %s seconds ---" % (end_time - start_time))
    plt.imshow(maskedImg)
    plt.show()

def color_thresh(img):
    """
    Convert RGB to HSL and threshold to binary image using S channel
    """
    # 1. Convert the image from RGB to HSL
    # 2. Apply threshold on S channel to get binary image
    # Hint: threshold on H to remove green grass
    blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    # 1 - Convert image from RGB to HSL
    hsl_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

    # 2 - Threshold L channel for white-looking lines
    # white_filter = cv2.inRange(hsl_img[:, :, 1], 215, 255)

    # Brown Filter Iteration 1
    '''
    brown_filter = cv2.inRange(hsl_img[:, :, 0], 20, 40)  # hue threshold for yellow-brown
    brown_filter2 = cv2.inRange(hsl_img[:, :, 0], 40, 60)  # hue threshold for reddish-brown

    # combine the two thresholds using bitwise OR
    brown_filter = cv2.bitwise_or(brown_filter, brown_filter2)

    # apply saturation threshold to filter out non-brown colors
    brown_filter = cv2.bitwise_and(brown_filter, cv2.inRange(hsl_img[:, :, 1], 20, 60))  # saturation threshold
    brown_filter = cv2.bitwise_and(brown_filter, cv2.inRange(hsl_img[:, :, 2], 20, 50))  # lightness threshold
    '''

    #Brown Filter Iteration 2

    # hue threshold with a range of +/- 10
    brown_filter_hue = cv2.inRange(hsl_img[:, :, 0], 46, 66)

    # saturation threshold with a range of +/- 10%
    brown_filter_sat = cv2.inRange(hsl_img[:, :, 1], 78, 98)

    # lightness threshold with a range of +/- 5%
    brown_filter_light = cv2.inRange(hsl_img[:, :, 2], 5, 15)

    # combine the three thresholds using bitwise AND
    brown_filter = cv2.bitwise_and(brown_filter_hue, brown_filter_sat)
    brown_filter = cv2.bitwise_and(brown_filter, brown_filter_light)



    # if ROSBAG_3:
    #     yellow_low = np.array([25, 0, 90])
    #     yellow_high = np.array([35, 255, 255])
    #     white_filter1 = cv2.inRange(hsl_img[:, :, 1], 150, 255)
    #     white_filter = white_filter1
    #
    #     yellow_filter = cv2.inRange(hsl_img, yellow_low, yellow_high)
    #     yw_mask = cv2.bitwise_or(yellow_filter, white_filter) / 255  # Convert 0-255 to 0-1
    #
    #     dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    #     yw_mask = cv2.dilate(yw_mask, dilate_element)
    #     return yw_mask

    # 3 - Dilate area a little to coincide with sobel output later down the pipeline
    yw_mask = brown_filter / 255  # Convert 0-255 to 0-1

    dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    yw_mask = cv2.dilate(yw_mask, dilate_element)

    ####

    return yw_mask


def divotDetectorGrayBin():
    rawImg = cv2.imread("pictures\Divot1.jpg")

    blur = cv2.GaussianBlur(rawImg, (8001, 8001), cv2.BORDER_DEFAULT)
    plt.imshow(blur)
    plt.show()
    grayImg = rgb2gray(blur)
    binarized_gray = (grayImg > 0.7) * 1
    plt.imshow(binarized_gray, cmap='gray')
    plt.show()
def coffeeDetector():
    coffee = data.coffee()
    gray_coffee = rgb2gray(coffee)

    # Setting the plot size to 15,15
    # plt.figure(figsize=(15, 15))
    for i in range(10):
        # Iterating different thresholds
        binarized_gray = (gray_coffee > i * 0.1) * 1
        # plt.subplot(5, 2, i + 1)

        # Rounding of the threshold
        # value to 1 decimal point
        # plt.title("Threshold: >" + str(round(i * 0.1, 1)))

        # Displaying the binarized image
        # of various thresholds
        plt.imshow(binarized_gray, cmap='gray')
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
    blur = cv2.GaussianBlur(gray, (51, 51), cv2.BORDER_DEFAULT)  # 5 x 5 kernel size
    plt.imshow(blur)
    plt.show()

    # 3 - cv2.Sobel() to find derivatives
    ddepth = cv2.CV_16S
    gradient_x = cv2.Sobel(blur, ddepth, 1, 0)
    gradient_y = cv2.Sobel(blur, ddepth, 0, 1)

    # 4 - cv2.addWeighted() to combine the results
    combined = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0.0)

    # 5 - Convert each pixel to uint8 then apply threshold
    converted_uint8 = cv2.convertScaleAbs(combined)

    threshold = cv2.inRange(converted_uint8, thresh_min, thresh_max)
    threshold = threshold / 255

    ###

    return threshold


if __name__ == "__main__":
    greenDetector()
