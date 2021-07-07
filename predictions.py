# importing the dependencies
import cv2
import matplotlib.pyplot as plt
import numpy as np
from model import model
import os
import mimetype
import tensorflow as tf
from tensorflow import keras

H = 256
W = 256

print("Loading Trained Model...")
model = tf.load_model("model.h5")
print("Model Successfully loaded!")

# In this script we are using is making predictions on the model that we have already trained

path = input("Please enter the path for the file that you would like to test on: ")

valid_input = False
while not os.path.exists({}.format(path)):
    print("Error! The path that you entered does not exist.")
    print("Please try again.")
    path = input("Please enter the path for the file that you would like to test on: ")

    if os.path.exists({}.format(path)):
        name, extention = os.path.splitext(path)

        if extention == None:
            extention = mimetypes.guess_extention(path)

        elif extention == ".mp4" or extention == ".mov" or extention == ".mp3":
            print("Hmmm... The input is not in image format.")
            path = input("Please enter the path for the file that you would like to test on: ")

        elif extention == ".jpg" or extention == ".jpeg" or extention == ".png" :
            print("Success....")
            valid_input = True


counter = 0

if counter < 1 and valid_input == True:
    for data in os.listdir(path):
        counter += 1
        # reading in the image in by joining the path to get to the image
        image = cv2.imread(os.path.join(path, data))
        # we are resizing the image to the image_size that we set above
        image = cv2.resize(image, (H, W))

# convert the image to an array
img_array = np.array(image)

print(img_arrays.shape())

# expanding the dimensions to add the 3 at the end of the shape so that it can be passed in the network
#img_array = np.expand_dims(img_array, axis=0)

# make predictions on a new image that we pass into the network
predictions = model.predict(img_array)

# plotting the images
fig = plt.figure(15,15)
ax1, ax2 = plt.subplot(2)
ax1.imshow(image)
ax2.imshow(predictions)
plt.show()
cv2.waitKey(5)
