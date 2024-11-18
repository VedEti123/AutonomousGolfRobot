
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from divotDetector import divotDetector
from greenDetector import greenEdgeDetector
import RPi.GPIO as GPIO
from picamera import PiCamera
def rPiDeployment():


    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    #Use a variable for the Pin to use
    #If you followed my pictures, it's port 7 => BCM 4
    divotDet = 4
    greenDet = 5
    #Initialize your pin
    GPIO.setup(divotDet,GPIO.OUT)
    GPIO.setup(greenDet, GPIO.OUT)

    actionPerforming = False


    while True:

        # If action is being performed, wait for 5 seconds to ensure multiple actions aren't performed at once
        if actionPerforming:
            time.sleep(5)
            # Reset actionPerforming
            actionPerforming = False
            # Reset the GPIO pins
            GPIO.output(divotDet, GPIO.LOW)
            GPIO.output(greenDet, GPIO.LOW)

        # Get camera feed from raspberry pi camera
        startTime = time.perf_counter()

        # Convert camera feed capture to cv2 image
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.capture('/home/pi/image.jpg')

        img = cv2.imread('/home/pi/image.jpg')
        # Run divotDetector on given cv2 image
        divotFound = divotDetector(img)
        if divotFound:
            actionPerforming = True
            # Set GPIO pin to high
            GPIO.output(divotDet, GPIO.HIGH)
        # Run greenEdgeDetector on given cv2 image
        greenFound = greenEdgeDetector(img)
        if greenFound:
            actionPerforming = True
            # Set GPIO pin to high
            GPIO.output(greenDet, GPIO.HIGH)

        # Based on the results of the divotDetector and greenEdgeDetector, set the GPIO pins accordingly

        endTime = time.perf_counter()

        print(endTime - startTime)