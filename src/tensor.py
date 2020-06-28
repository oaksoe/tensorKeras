import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NB_CLASSES = 10
RESHAPED = 784

model = tf.keras.models.Sequential()

model.add(layers.Dense(NB_CLASSES, 
    input_shape=(RESHAPED,),
    kernal_initializer='zeros',
    name='dense_layer',
    activation='softmax'))