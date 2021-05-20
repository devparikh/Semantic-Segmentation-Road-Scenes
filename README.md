# Object-Detection-Segmentation
This is a project that uses semantic segmentation used to identify object's in a self-driving cars environment. The model that is used to create this project is called a U-Net architecture model and the dataset that was uses was Cityscape from kaggle. After doing data augmentations in preprocessing.py I was able to get 23,800 training images and 4000 testing images from the original 3000 training images and 457 testing images. For the training of the model I used different functions like learning rate scheduler, reduceonplateau and modelcheckpoint. The optimizer I used was Adam and the loss function is the tversky loss function which is a custom loss function.

![image](https://user-images.githubusercontent.com/47342287/117590294-c0556080-b0fc-11eb-80bb-3b0aaaf66944.png)
This is the U-Net architecture. 


# Dataset
https://www.kaggle.com/dansbecker/cityscapes-image-pairs


