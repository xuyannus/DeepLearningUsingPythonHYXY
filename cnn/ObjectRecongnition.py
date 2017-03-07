from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot

from cnn.DigitRecongnitionUsingMultiLayerNetwork import DataSet

from keras import backend as K
K.set_image_dim_ordering('th')


def load_data(data):
    # load (downloaded if needed)
    (data.X_train, data.y_train), (data.X_test, data.y_test) = cifar10.load_data()

    print data.X_train.shape
    print data.X_test.shape


def preprocess(data):

    # flatten 28*28 images to a 784 vector for each image
    data.num_pixels = data.X_train.shape[1] * data.X_train.shape[2]

    # reshape to be [samples][channels][width][height]
    data.X_train = data.X_train.astype('float32')
    data.X_test = data.X_test.astype('float32')

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

    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(data.num_classes, activation='softmax'))

    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model


def main():
    data = DataSet()

    load_data(data)

    preprocess(data)

    model = define_model(data)

    # Fit the model
    epochs = 25
    model.fit(data.X_train, data.y_train, validation_data=(data.X_test, data.y_test), nb_epoch=epochs, batch_size=32, verbose=2)

    # Final evaluation of the model
    scores = model.evaluate(data.X_test, data.y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    main()
