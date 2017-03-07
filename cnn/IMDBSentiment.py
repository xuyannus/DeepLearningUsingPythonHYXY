import numpy
from keras.datasets import imdb
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Embedding, Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from matplotlib import pyplot

from cnn.DigitRecongnitionUsingMultiLayerNetwork import DataSet


# fix the random seed
seed = 7


def set_context():
    # fix random seed for reproducibility
    numpy.random.seed(seed)


def load_data(data):
    # load (downloaded if needed) the MNIST dataset
    (data.X_train, data.y_train), (data.X_test, data.y_test) = imdb.load_data(nb_words=5000)

    max_words = 500
    data.X_train = sequence.pad_sequences(data.X_train, maxlen=max_words)
    data.X_test = sequence.pad_sequences(data.X_test, maxlen=max_words)

    # data.X = numpy.concatenate((data.X_train, data.X_test), axis=0)
    # data.y = numpy.concatenate((data.y_train, data.y_test), axis=0)
    #
    # # summarize size
    # print("Training data: ")
    # print(data.X.shape)
    # print(data.y.shape)
    #
    # # Summarize number of classes
    # print("Classes: ")
    # print(numpy.unique(data.y))
    #
    # # Summarize number of words
    # print("Number of words: ")
    # print(len(numpy.unique(numpy.hstack(data.X))))
    #
    # # Summarize review length
    # print("Review length: ")
    # result = map(len, data.X)
    # print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
    #
    # # plot review length as a boxplot and histogram
    # pyplot.subplot(121)
    # pyplot.boxplot(result)
    # pyplot.subplot(122)
    # pyplot.hist(result)
    # pyplot.show()


def define_NL_model():
    # create model
    model = Sequential()

    model.add(Embedding(5000, 32, input_length=500))
    model.add(Flatten())

    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model


def define_CNN_model():
    # create model
    model = Sequential()

    model.add(Embedding(5000, 32, input_length=500))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))

    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model


def main():
    data = DataSet()

    set_context()

    load_data(data)

    # model = define_NL_model()

    model = define_CNN_model()

    model.fit(data.X_train, data.y_train, validation_data=(data.X_test, data.y_test), nb_epoch=2, batch_size=128,
              verbose=1)

    # Final evaluation of the model
    scores = model.evaluate(data.X_test, data.y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    main()
