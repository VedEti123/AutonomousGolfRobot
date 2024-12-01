import time
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern


import os
def greenClassifier():
    directory = 'pictures'
    files = os.listdir(directory)
    print(files)
    newFiles = []
    for f in files:
        newFiles.append("pictures\\" + f)
    print(newFiles)

    greenEdgeClass = []
    noGreenEdgeClass = []
    for img in newFiles:
        rawImg = cv2.imread(img)
        classifier = greenEdgeDetector(rawImg)

        print(img, classifier)
        if classifier:
            greenEdgeClass.append(img)
        else:
            noGreenEdgeClass.append(img)
    print()
    print(greenEdgeClass)
    print(noGreenEdgeClass)

def greenEdgeDetector(rawImg = None):
    if rawImg is None:
        rawImg = cv2.imread("pictures\EdgeGreen4.jpeg")
    # plt.imshow(rawImg)
    # plt.show()
    # print(type(rawImg))
    # Apply Gabor Filters from a filter bank
    rawImg = cv2.GaussianBlur(rawImg, (5, 5), cv2.BORDER_DEFAULT)
    filters = create_gaborfilters()
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(rawImg, -1, kern)  # Apply filter to image
        plt.imshow(image_filter)
        plt.show()
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(rawImg, image_filter, rawImg)

    rawImg = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)
    print(rawImg)
    # lbpClassifier(rawImg)
    edgeFound = lbpClassifier(rawImg)

    return edgeFound

    # Train SVM Classfier on all pixel in the image for each filter in the filter bank


def create_gaborfilters():
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree

    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

# Function to extract LBP features from a region of interest (ROI)
def extract_lbp_features(image, radius=3, n_points=8*3):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype('float') / hist.sum()  # Normalize the histogram
    return hist

def lbpClassifier(image):
    # Load the image in grayscale

    height, width = image.shape
    # Define the regions of interest (ROIs)
    roi1 = image[0:height//2, : ] # Example region 1 (top-left corner)
    roi2 = image[height//2:, :]  # Example region 2 (second region)

    # Extract LBP features from each ROI
    features_roi1 = extract_lbp_features(roi1)
    features_roi2 = extract_lbp_features(roi2)

    # Compare the texture features (e.g., using chi-squared distance)
    chi_squared_distance = 0.5 * np.sum(
        ((features_roi1 - features_roi2) ** 2) / (features_roi1 + features_roi2 + 1e-10))

    print(f"Chi-squared distance between textures: {chi_squared_distance}")
    if chi_squared_distance > 0.02:
        return True

    roi1 = image[:, 0:width//2]  # Example region 1 (top-left corner)
    roi2 = image[:, width//2:]  # Example region 2 (second region)

    # Extract LBP features from each ROI
    features_roi1 = extract_lbp_features(roi1)
    features_roi2 = extract_lbp_features(roi2)

    # Compare the texture features (e.g., using chi-squared distance)
    chi_squared_distance = 0.5 * np.sum(
        ((features_roi1 - features_roi2) ** 2) / (features_roi1 + features_roi2 + 1e-10))
    print(f"Chi-squared distance between textures: {chi_squared_distance}")
    if chi_squared_distance > 0.02:
        return True
    # Plot the ROIs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(roi1, cmap='gray')
    plt.title('ROI 1')
    plt.subplot(1, 2, 2)
    plt.imshow(roi2, cmap='gray')
    plt.title('ROI 2')
    plt.show()
def colorGreenEdgeDetector(img):
    """
        Convert RGB to HSL and threshold to binary image using S channel
        """
    # 1. Convert the image from RGB to HSL
    # 2. Apply threshold on S channel to get binary image
    # Hint: threshold on H to remove green grass
    blur = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    # 1 - Convert image from RGB to HSL
    hsl_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

    shape = hsl_img.shape
    print(shape)
    numPixels = shape[0] * shape[1]

    # hue threshold with a range of +/- 10
    rough_filter_hue = cv2.inRange(hsl_img[:, :, 0], 88, 108)

    # saturation threshold with a range of +/- 10%
    rough_filter_sat = cv2.inRange(hsl_img[:, :, 1], 54, 74)

    # lightness threshold with a range of +/- 5%
    rough_filter_light = cv2.inRange(hsl_img[:, :, 2], 48, 58)

    # combine the three thresholds using bitwise AND
    edge_filter = cv2.bitwise_and(rough_filter_hue, rough_filter_sat)
    edge_filter = cv2.bitwise_and(edge_filter, rough_filter_light)

    print(edge_filter)
    # 3 - Dilate area a little to coincide with sobel output later down the pipeline
    yw_mask = edge_filter / 255  # Convert 0-255 to 0-1

    dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    maskedImg = cv2.dilate(yw_mask, dilate_element)
    print(maskedImg)
    ####
    xCoor, yCoor = np.where(maskedImg == 1)
    print(len(xCoor), numPixels)
    if len(xCoor) < numPixels * 0.75:
        print("Green Edge Detected")
        return True
    else:
        return False
    # return yw_mask
if __name__ == "__main__":
    #greenClassifier()
    rawImg = cv2.imread("EdgePictures\edge1.jpg")
    plt.imshow(rawImg)
    plt.show()
    # greenEdgeDetector(rawImg)
    colorGreenEdgeDetector(rawImg)