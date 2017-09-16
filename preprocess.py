from random import shuffle
import glob
import numpy as np
import tables
import cv2

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'dataset.hdf5'  # address to where you want to save the hdf5 file
train_path = 'training/*/*'
directories = glob.glob('training/*')
# read addresses and labels from the 'train' folder
addrs = glob.glob(train_path)
addrs.sort()
labels = []

for i in range (len(directories)):

    tmp = glob.glob(directories[i] + '/*')
    print(tmp)
    for j in range (len(tmp)):
        labels.append(i);


print("Number of images: " + str(len(addrs)))
print(labels)
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]
val_addrs = addrs[int(0.8*len(addrs)):int(0.9*len(addrs))]
val_labels = labels[int(0.8*len(addrs)):int(0.9*len(addrs))]
test_addrs = addrs[int(0.9*len(addrs)):]
test_labels = labels[int(0.9*len(labels)):]

img_dtype = tables.UInt8Atom()
data_shape = (0, 224, 224, 3)
hdf5_file = tables.open_file(hdf5_path, mode='w')

train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)


hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)

# a numpy array to save the mean of the images
mean = np.zeros(data_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    print(addr)

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change

    # save the image and calculate the mean so far
    train_storage.append(img[None])
    mean += img / float(len(train_labels))
# loop over validation addresses
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change

    # save the image
    val_storage.append(img[None])
# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change

    # save the image
    test_storage.append(img[None])
# save the mean and close the hdf5 file
hdf5_file.close()
