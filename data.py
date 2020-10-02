import tensorflow as tf
from tensorflow.keras.preprocessing import image
import random

def data_source(directory):
    datagen = image.ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(480, 640),
        batch_size=32,
        shuffle=True,
        class_mode='input')

    return generator

def augment(image):
    flipped_inputs = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(224, 224, 3))(image)
    contrasted = tf.keras.layers.experimental.preprocessing.RandomContrast(0.125, seed=random.randint(0,99))(flipped_inputs)
    zoomed = tf.keras.layers.experimental.preprocessing.RandomZoom(0.25, seed=random.randint(0, 99))(contrasted)
    translated = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.125, 0.12, seed=random.randint(0,99))(zoomed)
    cropped = tf.keras.layers.experimental.preprocessing.CenterCrop(224, 224)(translated)
    #zoomed_inputs = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)(rotated_inputs)


    return cropped

@tf.function
def augment_image(image):
    batch = tf.shape(image)[0]
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    image = tf.image.random_brightness(image, 0.25)
    ##if tf.random.uniform(()) > 0.9:
    ##    image = tf.image.rgb_to_grayscale(image)
    sf = .2*tf.random.uniform(())+0.9
    image = tf.image.random_crop(image, [batch, sf*360, sf*480, 3])
    image = tf.image.resize(image, [ 360, 480])
    ##return tf.image.central_crop(image, (224, 224))
    return tf.image.crop_to_bounding_box(image, 68, 128, 224, 224)
