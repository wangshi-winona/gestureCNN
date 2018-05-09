#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os


# We require this for Theano lib ONLY. Remove it for TensorFlow usage
from keras import backend as K
K.set_image_dim_ordering('th')

nb_filters=32
nb_conv=3
img_channels=1
img_rows=200
img_cols=200
nb_pool=2
nb_classes=4
batch_size=30
nb_epoch=15
defaultfname="new_weight.hdf5"
path = "./"

path2="./mytrain2"##for our trainning

# outputs
output = ["NOTHING", "PAPER","ROCK", "SCISSORS"]



#%%
def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist

def loadCNN(wf_index, fname):
    global get_output
    model = Sequential()

    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding='valid', data_format = 'channels_first',

                     input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Model summary
    model.summary()
    # Model conig details
    model.get_config()

    if wf_index >= 0:
        # Load pretrained weights
        print "loading ", fname
        model.load_weights(fname)

    print len(model.layers)
    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model, get_output


def initializers():
    imlist = modlistdir(path2)

    image1 = np.array(Image.open(path2 + '/' + imlist[0]))  # open one image to get size
    # plt.imshow(im1)

    m, n = image1.shape[0:2]  # get the size of the images
    total_images = len(imlist)  # get the 'total' number of images

    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2 + '/' + images).convert('L')).flatten()
                         for images in imlist], dtype='f')

    print immatrix.shape

    raw_input("Press any key")

    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label = np.ones((total_images,), dtype=int)

    samples_per_class = total_images / nb_classes
    print "samples_per_class - ", samples_per_class
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class

    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''

    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    (X, y) = (train_data[0], train_data[1])

    # Split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test


def trainModel(model):

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    ##visualizeHis(hist)

    ans = raw_input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = raw_input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname,overwrite=True)
    else:
        model.save_weights("newWeight.hdf5",overwrite=True)

if __name__ == "__main__":
    mod=loadCNN(-1, "")
    trainModel(mod)