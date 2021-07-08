import tensorflow as tf
import matplotlib.pyplot as plt
import os
from imutils.video import VideoStream 
import cv2
from model import model
import numpy as np
import mimetype

H = 256
W = 256

print("Loading Trained Model...")
model = tf.load_model("model.h5")
print("Model Successfully loaded!")

# In this script we are using is making predictions on the model that we have already trained

print("Here we are performing semantic segmentation on live video feed.")

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
        
        elif extention == ".jpg" or extention == ".jpeg" or extention == ".png" :
            print("Hmmm... The input is not in image format.")
            path = input("Please enter the path for the file that you would like to test on: ")

        elif extention == ".mp4" or extention == ".mov" or extention == ".mp3":
            print("Success....")
            valid_input = True

saving_path = "C:\\Users\\me\\Documents\\Object-Detection\\splitvideo"

video = cv2.VideoCapture(path)
counter = 0

while (cap.isOpened() and valid_input == True):
    # ret returns a boolean depending on whether or not there was a return
    ret, frame = video.read()

    if ret == False:
        break
    
    os.chdir(saving_path)
    cv2.imwrite("roadscene", str(counter), extention, frame)

    counter += 1 

video.release()
video.destroyAllWindows()

counter = 0

for data in os.listdir(saving_path):
    counter += 1
    # reading in the image in by joining the path to get to the image
    image = cv2.imread(os.path.join(saving_path, data))
    # we are resizing the image to the image_size that we set above
    image = cv2.resize(image, (H, W))

    # convert the image to an array
    img_array = np.array(image)

    # expanding the dimensions to add the 3 at the end of the shape so that it can be passed in the network
    #img_array = np.expand_dims(img_array, axis=0)

    # make predictions on a new image that we pass into the network
    predictions = model.predict(img_array)

    pred_image = np.array(predictions)

    width, height, channels = pred_image.shape()
    size = (width, height)
    seg_array.append(pred_image)

    output_frame = cv2.VideoWriter("output{}".format(counter), cv2.VideoWriter_fourcc(*extention), 60, size)
    output_frame.write(seg_array[data])

    cv2.imshow("Input Video", frame)
    cv2.imshow("Output Video", output_frame)

    print("Press the letter 'e' on your keyword to close the window.")

    if cv2.waitKey(25) & 0xFF == ord("e"):
        break
    
output_frame.release()
