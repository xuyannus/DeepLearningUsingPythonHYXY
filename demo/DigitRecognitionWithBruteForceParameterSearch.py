# Plot ad hoc mnist instances
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
# import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
import numpy

# fix the random seed
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

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


def load_photos():
    # load (downloaded if needed) the MNIST dataset
    photos = DataSet()
    (photos.X_train, photos.y_train), (photos.X_test, photos.y_test) = mnist.load_data()

    print "photos.X_train.shape:", photos.X_train.shape
    print "photos.X_test.shape:", photos.X_test.shape

    return photos


def preprocess(photos):
    # flatten 28*28 images to a 784 vector for each image
    photos.num_pixels = photos.X_train.shape[1] * photos.X_train.shape[2]
    photos.X_train = photos.X_train.reshape(photos.X_train.shape[0], photos.num_pixels).astype('float32')
    photos.X_test = photos.X_test.reshape(photos.X_test.shape[0], photos.num_pixels).astype('float32')

    # normalize inputs from 0-255 to 0-1
    photos.X_train = photos.X_train / 255.0
    photos.X_test = photos.X_test / 255.0

    # one hot encode outputs
    photos.y_train = np_utils.to_categorical(photos.y_train)
    photos.y_test = np_utils.to_categorical(photos.y_test)
    photos.num_classes = photos.y_test.shape[1]


def create_a_basic_neural_network(photos):
    # create model
    nn_model = Sequential()
    nn_model.add(Dense(photos.num_pixels, input_dim=photos.num_pixels, init='normal', activation='relu'))
    nn_model.add(Dense(photos.num_classes, init='normal', activation='softmax'))

    # Compile model
    nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn_model


def create_a_brute_force_nn(photos):

    # create a baseline nn model
    model = KerasClassifier(build_fn=create_a_basic_neural_network, photos=photos, verbose=0)

    # grid search epochs, batch size and optimizer
    optimizers = ['rmsprop', 'adam']
    init_distribution = ['glorot_uniform', 'normal', 'uniform']
    epochs = [10, 20]
    batches = [50, 100, 200]
    param_grid = dict(optimizer=optimizers, init=init_distribution, nb_epoch=epochs, batch_size=batches)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
    return grid


def main():

    # Step 1: download the data from MNIST dataset
    photos = load_photos()

    # Step 2: format transform for NN
    preprocess(photos)

    # Step 3: create a NN model
    model = create_a_brute_force_nn(photos)

    # Step 4: Model training
    brute_force_results = model.fit(photos.X_train, photos.y_train)
    print("Best: %f using %s" % (brute_force_results.best_score_, brute_force_results.best_params_))

    # Step 5: Final evaluation of the model
    scores = model.evaluate(photos.X_test, photos.y_test, verbose=2)

    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    main()
