
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__


data_gen = ImageDataGenerator(rescale=1./255)
dataset = data_gen.flow_from_directory('archive/flowers',
                                       target_size=(64, 64),
                                       batch_size=32,
                                       class_mode='categorical')
dataset.classes