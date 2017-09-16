
import tables

import keras
from keras.models import Model, Sequential # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


num_classes = 16

hdf5_path = 'dataset.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='r')
data_num = hdf5_file.root.train_img.shape[0]


train_X = np.array(hdf5_file.root.train_img)
train_Y = np.array(hdf5_file.root.train_labels)

val_X = np.array(hdf5_file.root.val_img)
val_Y = np.array(hdf5_file.root.val_labels)
train_Y = np_utils.to_categorical(train_Y, num_classes) # One-hot encode the labels
val_Y = np_utils.to_categorical(val_Y, num_classes) # One-hot encode the labels

hdf5_file.close()

batch_size = 10 # in each iteration, we consider 32 training examples at once
num_epochs = 300 # we iterate 200 times over the entire training set
kernel_size = 6 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

model = Sequential()

inp = Input(shape=(224, 224, 3)) # depth goes last in TensorFlow back-end (first in Theano)
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='sgd', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

valgen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)



# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_X)
valgen.fit(val_X)
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_X, train_Y, batch_size=10),
                    steps_per_epoch=len(train_X) / 10, epochs=num_epochs, validation_data=valgen.flow(val_X, val_Y), validation_steps=len(train_Y) / 10)

model.save('model.h5')
