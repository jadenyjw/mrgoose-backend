import keras

import tables
import numpy as np

from keras.models import Sequential

epochs = 100

hdf5_path = 'dataset.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='r')
data_num = hdf5_file.root.train_img.shape[0]


train_X = np.array(hdf5_file.root.train_img)
train_Y = np.array(hdf5_file.root.train_labels)

val_X = np.array(hdf5_file.root.val_img)
val_Y = np.array(hdf5_file.root.val_labels)
hdf5_file.close()

model = Sequential()




model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_X)



# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_X, train_Y, batch_size=10),
                    steps_per_epoch=len(train_X) / 32, epochs=epochs)
