import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import cv2
import imghdr

data_dir = 'dataset'
image_extension = ['jpg', 'jpeg', 'png', 'bmp']

# img = cv2.imread(os.path.join('dataset', 'Happy', '12-Things-a-Happy-Person-Does-Without-Realizing-It.jpg'))
#
from matplotlib import pyplot as plt
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)

        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_extension:
                print('Image not in extension list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            #os.remove(image_path)

import numpy as np

data = tf.keras.utils.image_dataset_from_directory('dataset')
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()
# print(batch[0].shape)

fig, ax = plt.subplots(ncols = 4, figsize = (20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()