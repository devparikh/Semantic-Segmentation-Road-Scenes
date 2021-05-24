# importing the dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
from model import *
import os
import tensorflow as tf
from tensorflow import keras

# In this script we are using is making predictions on the model that we have already trained

val_path = "C:\\Users\\me\\Documents\\Object-Detection\\cityscapes_data\\val"

counter = 0

if counter < 1:
    for data in os.listdir(val_path):
        counter += 1
        # reading in the image in by joining the path to get to the image
        image = cv2.imread(os.path.join(val_path, data))
        # we are resizing the image to the image_size that we set above
        image = cv2.resize(image, (image_size, image_size))

# data augmentation on the image
image = cv2.rotate(image, cv2.ROTATE_180)
image = cv2.flip(image, -1)

# convert the image to an array
img_array = np.array(image)

# load in the pre-trained model
model = keras.load_model("model.h5")

img_array.shape()

# make predictions on a new image that we pass into the network
predictions = model.predict(img_array)

# plotting the images
fig = plt.figure(5,5)
fig, ax1, ax2 = plt.subplot(2)
ax1.imshow(image)
ax2.imshow(predictions)
plt.show()