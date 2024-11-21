
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from divotDetector import divotDetector
from greenDetector import greenEdgeDetector
import RPi.GPIO as GPIO
import gpiod
from picamera2 import Picamera2, Preview
def rPiDeployment():


    # GPIO.setmode(GPIO.BCM)
    # GPIO.setwarnings(False)
    
    chip = gpiod.Chip('gpiochip4')

    #Use a variable for the Pin to use
    #If you followed my pictures, it's port 7 => BCM 4
    divotPin = 20
    greenPin = 21
    #Initialize your pin
    #GPIO.setup(divotDet,GPIO.OUT)
    #GPIO.setup(greenDet, GPIO.OUT)ls 
    
    divotGPIO = chip.get_line(divotPin)
    greenGPIO = chip.get_line(greenPin)
    
    divotGPIO.request(consumer = 'my_gpio', type = gpiod.LINE_REQ_DIR_OUT)
    greenGPIO.request(consumer = 'my_gpio', type = gpiod.LINE_REQ_DIR_OUT)
    
    
    camera = Picamera2()
    config = camera.create_still_configuration()
    camera.configure(config)
    camera.start()
    camera.awb_mode = 'daylight'

    time.sleep(2)

    actionPerforming = False
    divotGPIO.set_value(1)

    while True:

        # If action is being performed, wait for 5 seconds to ensure multiple actions aren't performed at once
        if actionPerforming:
            time.sleep(5)
            # Reset actionPerforming
            actionPerforming = False
            # Reset the GPIO pins
            #GPIO.output(divotDet, GPIO.LOW)
            #GPIO.output(greenDet, GPIO.LOW)
            divotGPIO.set_value(0)
            greenGPIO.set_value(0)
        # Get camera feed from raspberry pi camera
        startTime = time.perf_counter()
        print("Hello")
        # Convert camera feed capture to cv2 image)
        camera.resolution = (640, 480)
        #camera.capture_file('/home/codaero/image.jpg')
        camera.capture_file("cameraInput.jpg")
        #config = camera.create_preview_configuration()
        #camera.configure(config)
        
        #camera.capture_file('image.jpg')
        #rawImg = camera.capture_array("main")
        time.sleep(10)
        

        img = cv2.imread('cameraInput.jpg')
        # Run divotDetector on given cv2 image
        
        print("Hello")
        '''
        divotFound = divotDetector(img)
        if divotFound[0]:
            actionPerforming = True
            # Set GPIO pin to high
            #GPIO.output(divotDet, GPIO.HIGH)
            divotGPIO.set_value(1)
        # Run greenEdgeDetector on given cv2 image
        greenFound = greenEdgeDetector(img)
        if greenFound:
            actionPerforming = True
            greenGPIO.set_value(1)
            # Set GPIO pin to high
            #GPIO.output(greenDet, GPIO.HIGH)

        # Based on the results of the divotDetector and greenEdgeDetector, set the GPIO pins accordingly

        endTime = time.perf_counter()

        print(endTime - startTime)
        '''

if __name__=="__main__":
    rPiDeployment()
