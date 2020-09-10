import tensorflow as tf
import numpy as np
print(tf.__version__)

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras import regularizers

# For naming log files
from datetime import datetime


def simple():
    input_img = Input((224, 224, 3))
    conv1 = Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='valid')(input_img)
    pool1 = MaxPool2D((2,2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(pool1)
    pool2 = MaxPool2D((2,2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(pool2)
    pool3 = MaxPool2D((2,2))(conv3)
    conv4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(pool3)
    pool4 = MaxPool2D((3,3))(conv4)

    flat = Flatten()(pool4)
    dense1 = Dense(128, activation='relu')(flat)
    out = Dense(64)(dense1)
    mod = models.Model(inputs = input_img, outputs = out)
    return mod
    
