import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import imghdr

data_dir = 'data'
image_extension = ['jpg', 'jpeg', 'png', 'bmp']
