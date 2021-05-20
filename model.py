import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization, UpSampling2D, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping,  LearningRateScheduler
from tensorflow.python.ops.numpy_ops.np_math_ops import concatenate
from preprocessing import *
from loss import *
from tensorflow.keras.optimizers import Adam
from keras.models import Model

# Here we are going to create the U-Net model
# Contracting section of U-Net

# There are 4 contracting blocks each has 3x3 convolutional operations with and then you can add the batch normalization layer along with a ReLU activatation function and then 2x2 Maxpooling layer
model_input = Input(training_data)

# block # 1
convolutional1 = Conv2D(64, (3,3), activation="relu")(model_input)
batchnorm1 = BatchNormalization()(convolutional1)
convolutional2 = Conv2D(64, (3,3), activation="relu")(convolutional1)
batchnorm2 = BatchNormalization()(convolutional2)
maxpool1 = MaxPooling2D((2,2), stride=2)(convolutional2)

# block # 2
convolutional3 = Conv2D(128, (3,3), activation="relu")(maxpool1)
batchnorm3 = BatchNormalization()(convolutional3)
convolutional4 = Conv2D(128, (3,3), activation="relu")(convolutional3)
batchnorm4 = BatchNormalization()(convolutional4)
maxpool2 = MaxPooling2D((2,2), stride=2)(convolutional4)

# block # 3
convolutional5 = Conv2D(256, (3,3), activation="relu")(maxpool2)
batchnorm5 = BatchNormalization()(convolutional5)
convolutional6 = Conv2D(256, (3,3), activation="relu")(convolutional5)
batchnorm6 = BatchNormalization()(convolutional6)
maxpool3 = MaxPooling2D((2,2), stride=2)(convolutional6)

# block # 4
convolutional7 = Conv2D(512, (3,3), activation="relu")(maxpool3)
batchnorm7 = BatchNormalization()(convolutional1)
convolutional8 = Conv2D(512, (3,3), activation="relu")(convolutional1)
batchnorm8 = BatchNormalization()(convolutional2)
maxpool4 = MaxPooling2D((2,2), stride=2)


# bottleneck part of U-Net
upconv = UpSampling2D(size=(2,2))(convolutional8)
convolutional9 = Conv2D(1024, (3,3), activation="relu")(maxpool4)
batchnorm9 = BatchNormalization()(convolutional9)
convolutional10 = Conv2D(1024, (3,3), activation="relu")(convolutional9)
batchnorm10 = BatchNormalization()(convolutional10)


# the expansion part of U-Net
# block 1
upconv2 = UpSampling2D(size=(2,2))(convolutional10)
merge1 = concatenate([upconv2,convolutional4], axis=3) 
convolutional11 = Conv2D(512, (3,3), activation="relu")(upconv)
batchnorm11 = BatchNormalization()(convolutional11) 
convolutional12 = Conv2D(512, (3,3), activation="relu")(convolutional11)
batchnorm12 = BatchNormalization()(convolutional12) 
upconv2 = UpSampling2D(size=(2,2))(convolutional12)

# block 2
upconv3 = UpSampling2D(size=(2,2))(convolutional12)
merge2 = concatenate([upconv3,convolutional3], axis=3) 
convolutional13 = Conv2D(256, (3,3), activation="relu")(upconv2)
batchnorm13 = BatchNormalization()(convolutional13)
convolutional14 = Conv2D(256, (3,3), activation="relu")(convolutional13)
batchnorm14 = BatchNormalization()(convolutional14)


# block 3
upconv4 = UpSampling2D(size=(2,2))(convolutional14)
merge3 = concatenate([upconv4,convolutional2], axis=3) 
convolutional15 = Conv2D(128, (3,3), activation="relu")(upconv3)
batchnorm15 = BatchNormalization()(convolutional15)
convolutional16 = Conv2D(128, (3,3), activation="relu")(convolutional15)
batchnorm16 = BatchNormalization()(convolutional16)
upconv4 = UpSampling2D(size=(2,2))(convolutional16)

# final layer
upconv5 = UpSampling2D(size=(2,2))(convolutional16)
merge4 = concatenate([upconv5,convolutional1], axis=3)
convolutional17 = Conv2D(64, (3,3), activation="relu")(upconv4)
batchnorm17 = BatchNormalization()(convolutional17)
convolutional18 = Conv2D(64, (3,3), activation="relu")(convolutional17)
batchnorm18 = BatchNormalization()(convolutional18)
convolutional19 = Conv2D(64, (3,3), activation="relu")(convolutional18)
batchnorm18 = BatchNormalization()(convolutional18)
convolutional20 = Conv2D(64, (1,1), activation="sigmoid")(convolutional19)

# we are creating checkpoints for the model
epochs = 30
learning_rate = 0.1
model_filepath = "C:\\Users\\me\\Documents\\Object-Detection"

model_checkpoint = ModelCheckpoint(filepath=model_filepath,
                                   save_weights_only=True,
                                   monitor="val_accuracy",
                                   verbose=0,
                                   save_best_only=True,
                                   mode="max")

# We are reducing the learning rate depending on the minimizing of the loss function
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1,
                              patience=3, verbose=0, min_lr=0.0001)

# using the earlystopping function to do earlystopping in the occurance of overfitting
earlystopping = EarlyStopping(monitor="val_accuracy", 
                              patience=0, 
                              verbose=0,
                              min_delta=0,
                              mode="max",
                              baseline=None,
                              restore_best_weights=True)

# here we are going to do the learning rate scheduler
def learning_rate_scheduler(epoch, learning_rate):
    if epoch < 8:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)

learning_rate_scheduler(epochs, learning_rate)

learningratescheduler = LearningRateScheduler(learning_rate_scheduler, verbose=0)

model = Model(input=model_input, output=convolutional20)

# using the optimizer Adam and the loss function tversky loss from the loss.py script
model.compile(optimizer = Adam(1e-5), 
              loss=[tversky_loss], 
              matrics=["accuracy"])

model.fit(training_data, epochs=epochs, validation_data=validation_data, callbacks=[learningratescheduler,earlystopping,reduce_lr,model_checkpoint])
