# Plot ad hoc mnist instances
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

import numpy

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
    plt.subplot(221)
    plt.imshow(data.X_train[0], cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(data.X_train[1], cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(data.X_train[2], cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(data.X_train[3], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()


def preprocess(data):

    # flatten 28*28 images to a 784 vector for each image
    data.num_pixels = data.X_train.shape[1] * data.X_train.shape[2]
    data.X_train = data.X_train.reshape(data.X_train.shape[0], data.num_pixels).astype('float32')
    data.X_test = data.X_test.reshape(data.X_test.shape[0], data.num_pixels).astype('float32')

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
    model.add(Dense(data.num_pixels, input_dim=data.num_pixels, init='normal', activation='relu'))
    model.add(Dense(data.num_classes, init='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
