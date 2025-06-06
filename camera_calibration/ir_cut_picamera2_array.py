######## PiCamera2/OpenCV picture taker for the RPI IR-Cut camera #########

# Author: Petros626
# forked from/oriented on: https://github.com/EdjeElectronics/Image-Dataset-Tools/blob/main/PictureTaker/PictureTaker.py
# Date: 19.04.2023
# Description: 
# This program takes pictures (.png format - 95% quality and 0 compression) from a the RPi IR-Cut 
# camera and saves them in the specified directory. The default directory is 'images' and the
# default resolution is 1920x1080. Additionally lens undistortion with camera calibration algorithm are
# implemented.

# Example usage to save images in a directory named images at 1920x1080 resolution:
# python3 run_camera_config.py --imgdir=images --res=1920x1080

# This code is based off the Picamera2 library examples at:
# https://github.com/raspberrypi/picamera2/tree/a9f7a7d0bac726ab9b3f366ff461ddd62e885f40/examples
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# https://betterprogramming.pub/how-to-calibrate-a-camera-using-python-and-opencv-23bab86ca194

from picamera2 import Picamera2
from os import getcwd, path, makedirs
from argparse import ArgumentParser
from sys import stdin, exit
from libcamera import controls
from json import load
from numpy import array, float64
from cv2 import (
    cvtColor, imwrite, COLOR_RGBA2RGB, undistort, imshow, getOptimalNewCameraMatrix, waitKey, destroyAllWindows,
    namedWindow, WINDOW_NORMAL, startWindowThread, resizeWindow, moveWindow
)

#### Parser and safety requests #####
# Fetch script arguments
parser = ArgumentParser()
parser.add_argument("--imgdir", help = "Folder where the taken images get saved. If you not specify there will be created one automatically.",
                    default = "images")
parser.add_argument("--res", help = "Required resolution in WxH. To avoid erros find out about the supported resolutions of your camera model.",
                       default = "640x480")

args = parser.parse_args()
dirname = args.imgdir

# Check if resolution is specified correctly
if not "x" in args.res:
    print("Specify resolution with x as WxH. (Example: 1920x1080).")
    exit()
imgW, imgH = map(int, args.res.split("x"))

# Create a folder, if it doesn't exist
cwd = getcwd()
dirpath = path.join(cwd,dirname)
if not path.exists(dirpath):
    makedirs(dirpath)
    
# Prevent taken frame overwriting
imgnum  = 1
key_flag = False

while 1:
    filename = f"{dirname}_{imgnum}.png"
    if not path.exists(path.join(dirpath, filename)):
        break
    imgnum +=1

#### Initialize camera #####
# Load the tuning for the RPi IR-Cut camera.
# Renamed the original file (ov5647.json) to custom.
tuning_file = Picamera2.load_tuning_file("ov5647.json")
# Call picamera2 constructor and pass loaded tuning file.
picam2 = Picamera2(tuning=tuning_file)
# Set options for saving images
picam2.options["quality"] = 95 # best quality
picam2.options["compress_level"] = 0 # no compression

# Print the hints for the user
print("\n##############################")
print("### Image taker calibrated ###")
print("##############################")
print("For help run the script with the '--help' option.")
print("\nPress 'p' to take an image, they will be saved in the '{}' folder.".format(dirname))
print("To quit the application press 'q'.\n")

# Create preview configuration with denoising
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (imgW, imgH)}, controls={"NoiseReductionMode":controls.draft.NoiseReductionModeEnum.HighQuality}))
picam2.start()

# Setup the preview Window with OpenCV (PiCamera2 not compatible)
# Set size, position and window size
winname = "Calibrated (undistorted) Image taker"
namedWindow(winname, WINDOW_NORMAL)
resizeWindow(winname, 1000, 900)
moveWindow(winname, 915, 72)
startWindowThread()


try:
    for i in range(10):
            key = waitKey(1000)
            if key == ord('q'):
                break

            # request - faster?
            #request = picam2.capture_request()
            #array = request.make_array("main")
            
            #2 direct capture -faster ?
            array = picam2.capture_array("main")
            new_cv_img = cvtColor(array, COLOR_RGBA2RGB)
            h, w = new_cv_img.shape[:2]
            # hold the default image size (not cropping)
            #x, y, w, h = roi
            #dst = dst[y:y+h, x:x+w]
            
            imshow(winname, new_cv_img)

            filename = "".join([dirname, "_", str(imgnum), ".png"])
            savepath = path.join(dirpath, filename)
            imwrite(savepath, new_cv_img)
            print("\rOpenCV image saved from request -> {}".format(filename))
            imgnum += 1
finally:
    destroyAllWindows()
    picam2.stop()
    picam2.close()
