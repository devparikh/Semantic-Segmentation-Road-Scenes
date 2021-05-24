# importing the dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
from model import model
import os
import tensorflow as tf
from tensorflow import keras

print("Loading Trained Model...")
model = keras.load_model("model.h5")
print("Model Successfully loaded!")

# In this script we are using is making predictions on the model that we have already trained

path = input("Please enter the path for the file that you would like to test on: ")

while not os.path.exists({}.format(path)):
    print("Error! The path that you entered does not exist.")
    print("Please try again.")
    path = input("Please enter the path for the file that you would like to test on: ")

counter = 0

if counter < 1:
    for data in os.listdir(path):
        counter += 1
        # reading in the image in by joining the path to get to the image
        image = cv2.imread(os.path.join(val_path, data))
        # we are resizing the image to the image_size that we set above
        image = cv2.resize(image, (image_size, image_size))

# data augmentation on the image
image = cv2.rotate(image, cv2.ROTATE_180)
image = cv2.flip(image, -1)

# converting to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

norm_image = np.zeros(256,256)
image = cv2.normalize(image, norm_image, 0, 255, cv2.NORM_MINMAX)

# convert the image to an array
img_array = np.array(image)

# make predictions on a new image that we pass into the network
predictions = model.predict(img_array)

# plotting the images
fig = plt.figure(5,5)
fig, ax1, ax2 = plt.subplot(2)
ax1.imshow(image)
ax2.imshow(predictions)
plt.show()
cv2.waitKey(5)
