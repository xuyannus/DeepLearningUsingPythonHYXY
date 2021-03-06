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
num_pixels = 28 * 28
num_classes = 10


class DataSet:
    def __init__(self,):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


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
    # flatten 28*28 digits to a 784 vector for each image
    photos.X_train = photos.X_train.reshape(photos.X_train.shape[0], num_pixels).astype('float32')
    photos.X_test = photos.X_test.reshape(photos.X_test.shape[0], num_pixels).astype('float32')

    # normalize inputs from 0-255 to 0-1
    photos.X_train = photos.X_train / 255.0
    photos.X_test = photos.X_test / 255.0

    # one hot encode outputs
    photos.y_train = np_utils.to_categorical(photos.y_train)
    photos.y_test = np_utils.to_categorical(photos.y_test)


def create_a_basic_neural_network(optimizer='rmsprop', init_distribution='glorot_uniform', activation_fun='relu'):
    # create model
    nn_model = Sequential()
    nn_model.add(Dense(num_pixels, input_dim=num_pixels, init=init_distribution, activation=activation_fun))
    nn_model.add(Dense(num_classes, init='normal', activation='softmax'))

    # Compile model
    nn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return nn_model


def create_a_brute_force_nn():

    # create a baseline nn model
    model = KerasClassifier(build_fn=create_a_basic_neural_network, verbose=0)

    # grid search
    # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # optimizers = ['SGD', 'rmsprop', 'Adagrad', 'Adadelta', 'adam', 'Adamax', 'Nadam']
    # weight_constraint = [1, 2, 3, 4, 5]
    # dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # neurons = [1, 5, 10, 15, 20, 25, 30]
    optimizers = ['rmsprop', 'adam']
    init_distribution = ['normal']
    activation_fun = ['relu', 'sigmoid']
    epochs = [10]
    batches = [50]
    param_grid = dict(optimizer=optimizers, init_distribution=init_distribution, activation_fun=activation_fun, nb_epoch=epochs, batch_size=batches)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
    return grid


def main():

    # Step 1: download the data from MNIST dataset
    photos = load_photos()

    # Step 2: format transform for NN
    preprocess(photos)

    # Step 3: create a NN model
    model = create_a_brute_force_nn()

    # Step 4: Model training
    brute_force_results = model.fit(photos.X_train, photos.y_train)
    print("Best: %f using %s" % (brute_force_results.best_score_, brute_force_results.best_params_))

    means = brute_force_results.cv_results_['mean_test_score']
    stds = brute_force_results.cv_results_['std_test_score']
    params = brute_force_results.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":
    main()
