import gpiod
import time

chip = gpiod.Chip('gpiochip4')


divotPin = 20
greenPin = 21
#Initialize your pin
#GPIO.setup(divotDet,GPIO.OUT)
#GPIO.setup(greenDet, GPIO.OUT)ls 

divotGPIO = chip.get_line(divotPin)
greenGPIO = chip.get_line(greenPin)

divotGPIO.request(consumer = 'my_gpio', type = gpiod.LINE_REQ_DIR_OUT)
greenGPIO.request(consumer = 'my_gpio', type = gpiod.LINE_REQ_DIR_OUT)


divotGPIO.set_value(1)

boole = True

while (1):
    if boole:
        greenGPIO.set_value(1)
    else:
        greenGPIO.set_value(0)
    boole = not boole
    time.sleep(2)