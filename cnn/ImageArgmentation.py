import os

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

from cnn.DigitRecongnitionUsingMultiLayerNetwork import DataSet


from keras import backend as K
K.set_image_dim_ordering('th')


def load_data(data):
    # load (downloaded if needed) the MNIST dataset
    (data.X_train, data.y_train), (data.X_test, data.y_test) = mnist.load_data()


def zca(data):
    # reshape to be [samples][pixels][width][height]
    data.X_train = data.X_train.reshape(data.X_train.shape[0], 1, 28, 28).astype('float32')
    data.X_test = data.X_test.reshape(data.X_test.shape[0], 1, 28, 28).astype('float32')

    # define data preparation
    datagen = ImageDataGenerator(zca_whitening=True)

    # fit parameters from data
    datagen.fit(data.X_train)

    # configure batch size and retrieve one batch of digits
    for X_batch, y_batch in datagen.flow(data.X_train, data.y_train, batch_size=9):
        # create a grid of 3x3 digits
        for i in range(0, 9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            # show the plot

        pyplot.show()
        break


def random_rotate(data):
    # define data preparation
    shift = 0.2
    datagen = ImageDataGenerator(rotation_range=90, width_shift_range=shift, height_shift_range=shift)

    # fit parameters from data
    datagen.fit(data.X_train)

    # configure batch size and retrieve one batch of digits
    for X_batch, y_batch in datagen.flow(data.X_train, data.y_train, batch_size=9):
        # create a grid of 3x3 digits
        for i in range(0, 9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))

        # show the plot
        pyplot.show()
        break


def save_to_file(data):

    # define data preparation
    datagen = ImageDataGenerator()

    # fit parameters from data
    datagen.fit(data.X_train)

    # configure batch size and retrieve one batch of digits
    os.makedirs('digits')

    for X_batch, y_batch in datagen.flow(data.X_train, data.y_train, batch_size=9, save_to_dir='digits', save_prefix='aug', save_format='png'):
        # create a grid of 3x3 digits
        for i in range(0, 9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))

        # show the plot
        # pyplot.show()
        break


def main():
    data = DataSet()

    load_data(data)

    zca(data)
    #
    # random_rotate(data)

    save_to_file(data)


if __name__ == "__main__":
    main()
