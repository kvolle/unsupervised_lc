import models as local_models
import data

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K

# Move these to new file
def cosine_distance(vects):
    # is a distance, so taking compliment of similarity
    x, y = vects
    cosine = tf.keras.layers.Dot(1, normalize=True)([x, y])
    return 1 - cosine

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


#def clr_loss(vects):
def distance_loss(y_true, y_pred):
    #x, y = vects
    x, y = tf.split(y_pred, 2, 1)
    batch = tf.shape(x)[0]
    if (batch == None):
        batch = 3
    dots = tf.matmul(x, y, transpose_b=True)
    identity = tf.eye(batch)
    comp = tf.linalg.band_part(tf.ones(shape=(batch, batch)), 2, 2) - identity
    true = identity - tf.multiply(identity, dots)
    false = tf.multiply(comp, dots)
    return K.sum(true) + K.sum(false)

# Define data source and declare input
train = data.data_source("./Data")
input_img = layers.Input(shape=(480, 640, 3), name="input_img")

# Augment each image twice
augmented_one = tf.keras.layers.Lambda(data.augment_image, output_shape=(None, 224, 224, 3))(input_img)
augmented_two = tf.keras.layers.Lambda(data.augment_image, output_shape=(None, 224, 224, 3))(input_img)

# Calculate the output projection
net = local_models.simple()
latent_one = net(augmented_one)
latent_two = net(augmented_two)

encodings = layers.concatenate([latent_one, latent_two], axis=1)
# Calculate the distance between outputs
#distance =  tf.keras.layers.Lambda(cosine_distance,
#                  output_shape=eucl_dist_output_shape)([latent_one, latent_two])
#distance2 = tf.keras.layers.Lambda(cosine_distance,
#                  output_shape=eucl_dist_output_shape)([latent_one, tf.random.shuffle(latent_two)])
#distance = tf.keras.layers.Lambda(clr_loss,
#                  output_shape=eucl_dist_output_shape)([latent_one, latent_two])

# Declare the model
model = models.Model(inputs=[input_img], outputs=[encodings, latent_one, latent_two])

# Print model summary
model.summary()
#tf.keras.utils.plot_model(model, "test.png")

## Define the loss function
## (This should also be pulled to new file)
#def distance_loss(y_true, y_pred):
#    return tf.keras.backend.sum(tf.keras.backend.abs(y_pred))

# Compile the model
model.compile(optimizer=optimizers.Adam(), loss=[distance_loss, None, None])
#model.compile(optimizer=optimizers.Adam(), loss=[MeanSquaredError, MeanSquaredError, distance_loss])

# Model won't train, so predict
#x = next(train)
#z = model.predict(x)

# Train the model
EPOCHS = 5
model.fit(train, epochs=EPOCHS)

_, out1, out2 = model.predict(train)
matrix = np.matmul(out1[0:5,:], out2[0:5,:].T)
np.savetxt("./out_vec1.csv", out1[0:5,:], delimiter=",")
np.savetxt("./out_vec2.csv", out2[0:5,:], delimiter=",")
np.savetxt("./out_mat.csv", matrix, delimiter=",")
