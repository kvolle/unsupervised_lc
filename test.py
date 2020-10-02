import models as local_models
import data

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError
# Move these to new file
def cosine_distance(vects):
    # is a distance, so taking compliment of similarity
    x, y = vects
    cosine = tf.keras.layers.Dot(1, normalize=True)([x, y])
    return 1 - cosine

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# Define data source and declare input
train = data.data_source("./Data")
input_img = layers.Input(shape=(480, 640, 3), name="input_img")

# Augment each image twice
augmented_one = tf.keras.layers.Lambda(data.augment_image, output_shape=(None, 224, 224, 3))(input_img)
augmented_two = tf.keras.layers.Lambda(data.augment_image, output_shape=(None, 224, 224, 3))(input_img)


# Declare the model
model = models.Model(inputs=input_img, outputs=[augmented_one, augmented_two])

# Print model summary
model.summary()

# Define the loss function
# (This should also be pulled to new file)
def loss_fn(y_true, y_pred):
    return tf.keras.backend.mean(y_pred)

# Compile the model
model.compile(optimizer=optimizers.Adam(), loss=MeanSquaredError)#loss_fn)

# Model won't train, so predict
x = next(train)
z1, z2 = model.predict(x)
#print(x.shape)

import numpy as np
from matplotlib import image

z1 = np.clip(np.reshape(z1[0,:,:,:], (224, 224, 3)), 0., 1.)
z2 = np.clip(np.reshape(z2[0,:,:,:], (224, 224, 3)), 0., 1.)
#:image.imsave("x.png", np.reshape(x[0,:,:,:], (480, 640, 3)))
image.imsave("z1.png", z1)
image.imsave("z2.png", z2)
#image.imsave("diff.png", np.reshape(z1-z2, (224, 224, 3)))
print(np.mean(np.mean(np.mean(np.mean(z1-z2)))))
# Train the model
#EPOCHS = 10
#model.fit(train)
