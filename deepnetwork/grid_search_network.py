# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy


def set_context():
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)


def load_data():
    # load pima indians dataset
    dataset = numpy.loadtxt("../data/pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    return X, Y


# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init=init, activation='relu'))
    model.add(Dense(8, init=init, activation='relu'))
    model.add(Dense(1, init=init, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def findBestModel(X, Y):
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    # grid search epochs, batch size and optimizer
    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal']
    epochs = [100]
    batches = [20]

    param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def main():
    set_context()
    X, Y = load_data()
    findBestModel(X, Y)


if __name__ == "__main__":
    main()
