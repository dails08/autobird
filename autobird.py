import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from picamera import PiCamera
import time

# load tflite model
# load bird list

# if it's been less than an hour since last tweet, sleep for 1s and continue
# Wait ten seconds
# Take a picture
# resize picture
# load picture
# invoke model
# label and sort preds
# pull top five
# check if birds
# if bird, tweet and reset timer

interpreter = tf.lite.Interpreter("./models/inception_resnet_v2_2018_04_27/inception_resnet_v2.tflite")
interpreter.allocate_tensors()

# init camera
# set timer
# Check timer
# Take picture
camera = PiCamera()
camera.resolution = (1024, 768)
camera.start_preview()
# Camera warm-up time
time.sleep(2)
camera.capture('/data/foo.jpg')

birdpic = Image.open("./data/foo.jpg")


with open("./data/list_of_birds.txt") as f:
    birdnames = set([x.strip() for x in f.readlines()])

is_bird = False
for pred in list(labeled_preds_df.sort_values(by = "conf", ascending=False).iloc[:5,0]):
    if pred in birdnames:
        print("It's a bird!")
        is_bird = True
        break
        