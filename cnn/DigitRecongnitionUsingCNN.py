# Plot ad hoc mnist instances
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt

from keras.utils.visualize_util import plot

import numpy

from keras import backend as K
K.set_image_dim_ordering('th')


# fix the random seed
seed = 7


class DataSet:
    def __init__(self,):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.num_classes = 0
        self.num_pixels = 0


def set_context():
    # fix random seed for reproducibility
    numpy.random.seed(seed)


def load_data(data):
    # load (downloaded if needed) the MNIST dataset
    (data.X_train, data.y_train), (data.X_test, data.y_test) = mnist.load_data()


def plot_data(data):
    # plot 4 images as gray scale

    for index in range(0, 1000):
        plt.figure(figsize=(5, 5))
        plt.imshow(data.X_train[index], cmap=plt.get_cmap('gray'))
        plt.savefig('../demo/images/numbers_{}.png'.format(index))


def preprocess(data):

    # flatten 28*28 images to a 784 vector for each image
    data.num_pixels = data.X_train.shape[1] * data.X_train.shape[2]

    # reshape to be [samples][channels][width][height]
    data.X_train = data.X_train.reshape(data.X_train.shape[0], 1, 28, 28).astype('float32')
    data.X_test = data.X_test.reshape(data.X_test.shape[0], 1, 28, 28).astype('float32')

    # normalize inputs from 0-255 to 0-1
    data.X_train = data.X_train / 255
    data.X_test = data.X_test / 255

    # one hot encode outputs
    data.y_train = np_utils.to_categorical(data.y_train)
    data.y_test = np_utils.to_categorical(data.y_test)
    data.num_classes = data.y_test.shape[1]


def define_model(data):
    # create model
    model = Sequential()

    model.add(Convolution2D(30, 5, 5, input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(data.num_classes, init='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot(model, to_file='./DigitRecongnitionUsingCNN.png')

    return model


def main():
    data = DataSet()

    set_context()

    load_data(data)

    # plot_data(data)

    preprocess(data)

    model = define_model(data)

    # Fit the model
    model.fit(data.X_train, data.y_train, validation_data=(data.X_test, data.y_test), nb_epoch=10, batch_size=200,
              verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(data.X_test, data.y_test, verbose=0)

    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


if __name__ == "__main__":
    main()
