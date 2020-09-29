import tensorflow as tf
import numpy as np
print(tf.__version__)

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten, BatchNormalization, ReLU, UpSampling2D, concatenate, Activation
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

def conv_block(input_layer, filter_num):
  temp = Conv2D(filters=filter_num, kernel_size=3, padding='same')(input_layer)
  temp = BatchNormalization()(temp)
  temp = ReLU()(temp)
  return temp

def max_pool_arg(input_layer, size):
  output, argmax = tf.nn.max_pool_with_argmax(input_layer, size, strides=2, padding='SAME')
  return [output, argmax]

def unpool(input_layer, input_indices, scale_factor=2):
  ret = UpSampling2D(size=scale_factor)(input_layer)
  return ret

def autoencoder():
    n_block1 = 16
    n_block2 = 32
    n_block3 = 64

    input_img = Input((224, 224, 3))
    block_1 = conv_block(input_img, n_block1)
    block_2 = conv_block(block_1, n_block1)
    pool_1, indices_1 = max_pool_arg(block_2, 2)
    block_3 = conv_block(pool_1, n_block2)
    block_4 = conv_block(block_3, n_block2)
    pool_2, indices_2 = max_pool_arg(block_4, 2)
    block_5 = conv_block(pool_2, n_block3)
    block_6 = conv_block(block_5, n_block3)
    block_7 = conv_block(block_6, n_block3)
    pool_3, indices_3  = max_pool_arg(block_7, 2)
    latent = Flatten()(pool_3)
    unpool_1 = unpool(pool_3, indices_3, 2)
    deconv_block_1 = conv_block(unpool_1, n_block3)
    deconv_block_2 = conv_block(deconv_block_1, n_block3)
    deconv_block_3 = conv_block(deconv_block_2, n_block3)
    unpool_2 = unpool(deconv_block_3, indices_2, 2)
    deconv_block_4 = conv_block(unpool_2, n_block2)
    deconv_block_4 = concatenate([deconv_block_4, block_4])
    deconv_block_5 = conv_block(deconv_block_4, n_block2)
    unpool_3 = unpool(deconv_block_5, indices_1, 2)
    deconv_block_6 = conv_block(unpool_3, n_block1)
    deconv_block_6 = concatenate([deconv_block_6, block_1])
    deconv_block_7 = conv_block(deconv_block_6, n_block1)
    out = Activation('tanh')(conv_block(deconv_block_7, 3))
    mod = models.Model(inputs = input_img, outputs = [out, latent])
    return mod
    
