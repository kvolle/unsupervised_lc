import tensorflow as tf
from tensorflow.keras.preprocessing import image

def data_source(directory):
    datagen = ImageDataGenerator(rescale=1./255i)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(256, 256),
        batch_size=32,
        class_mode=None)
    return generator

def augment(image):
    flipped_inputs = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(224, 224, 3))(image)
    contrasted = tf.keras.layers.experimental.preprocessing.RandomContrast(0.125)(flipped_inputs)
    zoomed = tf.keras.layers.experimental.preprocessing.RandomZoom(0.25)(contrasted)
    translated = tf.keras.layers.experimental.preprocessing.RandomTranslate(0.125)(zoomed)
    cropped = tf.keras.layers.experimental.preprocessing.CenterCrop(224, 224)(translated)
    #zoomed_inputs = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)(rotated_inputs)


    return cropped
