import models as local_models
import data

import numpy as np
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


def clr_loss(vects):
    x, y = vects
    batch, feats = x.shape
    print(x.shape)
    dots = tf.matmul(x, y, transpose_b=True)
    identity = tf.eye(batch)
    comp = tf.linalg.band_part(tf.ones(shape=(batch, batch)), 1, 1) - identity
    true = tf.matmul(identity, dots)
    false = comp - tf.matmul(comp, dots)
    return true + false

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

# Calculate the distance between outputs
#distance =  tf.keras.layers.Lambda(cosine_distance,
#                  output_shape=eucl_dist_output_shape)([latent_one, latent_two])
#distance2 = tf.keras.layers.Lambda(cosine_distance,
#                  output_shape=eucl_dist_output_shape)([latent_one, tf.random.shuffle(latent_two)])
distance = tf.keras.layers.Lambda(clr_loss,
                  output_shape=eucl_dist_output_shape)([latent_one, latent_two])

# Declare the model
model = models.Model(inputs=[input_img], outputs=[distance, latent_one, latent_two])

# Print model summary
#model.summary()
#tf.keras.utils.plot_model(model, "test.png")

# Define the loss function
# (This should also be pulled to new file)
def distance_loss(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.abs(y_pred))

# Compile the model
model.compile(optimizer=optimizers.Adam(), loss=[distance_loss])
#model.compile(optimizer=optimizers.Adam(), loss=[MeanSquaredError, MeanSquaredError, distance_loss])

# Model won't train, so predict
#x = next(train)
#z = model.predict(x)

# Train the model
EPOCHS = 50
model.fit(train, epochs=EPOCHS)

_, out1, out2 = model.predict(train)
matrix = np.matmul(out1, out2.T)
np.savetxt("./out_vec1.csv", out1, delimiter=",")
np.savetxt("./out_vec2.csv", out2, delimiter=",")
np.savetxt("./out_mat.csv", matrix, delimiter=",")
