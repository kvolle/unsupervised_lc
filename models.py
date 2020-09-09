import tensorflow as tf
import numpy as np
print(tf.__version__)

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras import regularizers

# For naming log files
from datetime import datetime


def simple(input_tensor):
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(input_tensor)
    pool1 = MaxPool2D((2,2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(pool1)
    pool2 = MaxPool2D((2,2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='valid')(pool2)
    pool3 = MaxPool2D((2,2))(conv3)
    dense1 = Dense(128, activation='relu')(pool3)
    out = Dense(64)(dense1)
    return out
    
