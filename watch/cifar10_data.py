from keras.applications.resnet50 import ResNet50, preprocess_input
import keras
from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras

batch_size = 32
num_classes = 10
epochs = 100
Iterations = 64000

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train / 255.0
x_test = x_test / 255.0
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = models.Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(num_classes, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

datagen = ImageDataGenerator(rescale = 1./255.,
                            rotation_range = 40,
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True)

datagen.fit(x_train)
# train_data = datagen.flow(x_train, y_train, batch_size=batch_size)
train_data = datagen.flow(x_train, y_train, batch_size=len(x_train))

val_data = datagen.flow(x_test, y_test, batch_size=len(x_test))
iter_ = 0
for x_train_, y_train_ in train_data:
    x_test_, y_test_ = next(val_data)
    if iter_ % 100 == 0:
        model.fit(x_train_, y_train_,
            batch_size=batch_size,
            validation_data=(x_test_, y_test_))
        continue
    model.fit(x_train_, y_train_,
        batch_size=batch_size,
        validation_data=(x_test_, y_test_))
    iter_ = iter_ + 1
    if iter >= Iterations:
        break

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# history = model.fit(x_train, y_train, epochs=5, batch_size=batch_size, validation_data=(x_test, y_test))
'''
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.utils import np_utils
from keras import backend as K
import os
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras
# print(tf.__version__)

# exit()

batch_size = 32
num_classes = 10
epochs = 100
Iterations = 64000

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


base_model = ResNet50(include_top=False, weights=None,
             input_shape=(200, 200, 3))

X_train_new = np.array([imresize(x_train[i], (200, 200, 3)) for i in range(0, len(X_train))]).astype('float32')
resnet_train_input = preprocess_input(X_train_new)
# model = Sequential()
# model.add(base_model)
# model.add(keras.layers.Flatten())
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.BatchNormalization())
# dmodel.add(keras.layers.Dense(10, activation='softmax'))

# model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

datagen = ImageDataGenerator(rescale = 1./255.,
                            rotation_range = 40,
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagen.fit(x_train)
train_data = datagen.flow(x_train, y_train, batch_size=batch_size)

iter_ = 0
for x_train_, y_train_ in train_data:
    if iter_ % 100 == 0:
        model.fit(x_train_, y_train_,
            batch_size=batch_size,
            validation_data=(x_test, y_test))
        continue
    model.fit(x_train_, y_train_,
        batch_size=batch_size, verbose = 0)
    iter_ = iter_ + 1

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

'''



