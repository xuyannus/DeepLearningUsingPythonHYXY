from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot

from keras.utils.visualize_util import plot

from keras import backend as K
K.set_image_dim_ordering('th')


seed = 7
num_pixels = 32 * 32
num_classes = 10


class DataSet:
    def __init__(self,):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


def load_data():
    photos = DataSet()
    # load (downloaded if needed)
    (photos.X_train, photos.y_train), (photos.X_test, photos.y_test) = cifar10.load_data()

    print photos.X_train.shape
    print photos.X_test.shape

    return photos


def preprocess(photos):

    # normalize inputs from 0-255 to 0-1
    photos.X_train = photos.X_train.astype('float32') / 255
    photos.X_test = photos.X_test.astype('float32') / 255

    # one hot encode outputs
    photos.y_train = np_utils.to_categorical(photos.y_train)
    photos.y_test = np_utils.to_categorical(photos.y_test)


def create_a_deep_neural_network():
    # create model
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    plot(model, to_file='./DeepNeuralNetworkToLearnObjects.png', show_shapes=True)
    return model


def main():
    # Step 1: download the data from Canadian Institute for Advanced Research (CIFAR) dataset.
    photos = load_data()

    # Step 2: format transform for NN
    preprocess(photos)

    # Step 3: create a DNN model
    model = create_a_deep_neural_network()

    # Step 4: Model training
    # purposely reduce the training epoch for training
    model.fit(photos.X_train, photos.y_train, nb_epoch=5, batch_size=32, verbose=2)

    # Step 5: Final evaluation of the model
    scores = model.evaluate(photos.X_test, photos.y_test, verbose=0)

    print("Evaluation Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    main()
