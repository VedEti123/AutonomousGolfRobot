from picamera2 import Picamera2, Preview
import time

# Create Picamera2 instance
picam2 = Picamera2()

# Configure camera settings
config = picam2.create_still_configuration()
picam2.configure(config)

# Start camera preview (optional)

# Start the camera
picam2.start()

# Allow camera to warm up
time.sleep(2)

# Capture image and save it
picam2.capture_file("edge5.jpg")

# Stop the camera
picam2.stop()

# Stop the preview
picam2.stop_preview()