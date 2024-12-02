# import gpiod
# import time

# chip = gpiod.Chip('gpiochip4')


# divotPin = 20
# greenPin = 21
# #Initialize your pin
# #GPIO.setup(divotDet,GPIO.OUT)
# #GPIO.setup(greenDet, GPIO.OUT)ls 

# divotGPIO = chip.get_line(divotPin)
# greenGPIO = chip.get_line(greenPin)

# divotGPIO.request(consumer = 'my_gpio', type = gpiod.LINE_REQ_DIR_OUT)
# greenGPIO.request(consumer = 'my_gpio', type = gpiod.LINE_REQ_DIR_OUT)


# divotGPIO.set_value(1)

# boole = True

# while (1):
#     if boole:
#         greenGPIO.set_value(1)
#     else:
#         greenGPIO.set_value(0)
#     boole = not boole
#     time.sleep(2)



import gpiozero
import time


if __name__=="__main__":
    divotPin = 20
    greenPin = 21

    divotGPIO = gpiozero.DigitalOutputDevice(divotPin, initial_value=False)
    greenGPIO = gpiozero.DigitalOutputDevice(greenPin, initial_value=False)

    divotGPIO.on()
    greenGPIO.on()

    GPIO16 = gpiozero.DigitalOutputDevice(16, initial_value=False)
    GPIO16.on()

    print("Hello")
    while 1:
        pass

    print("d")