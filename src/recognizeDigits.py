import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Network and training parameters.
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10                 # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2          # how much TRAIN is reserved for VALIDATION

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

RESHAPED = 784

# X_train is 3-D array of 60000 * 28 * 28. After reshape, it becomes 2-D array of 60000 * 784.
# The final value of the array is between 0 to 255, after normalization
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize inputs to be within in [0, 1]
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One-hot encoding
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Build the model
model = tf.keras.models.Sequential()
model.add(layers.Dense(NB_CLASSES, 
    input_shape=(RESHAPED,),
    name='dense_layer',
    activation='softmax'))

# Compile the model
model.compile(optimizer='SGD',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)