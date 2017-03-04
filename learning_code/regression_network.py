import numpy
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# fix the random seed
seed = 7


def set_context():
    # fix random seed for reproducibility
    numpy.random.seed(seed)


def load_data():
    # load dataset
    dataframe = read_csv("../data/housing.csv", delim_whitespace=True, header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    Y = dataset[:, 13]
    return X, Y


def define_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, init='normal', activation='relu'))
    # model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def main():
    set_context()
    X, Y = load_data()

    # # evaluate model
    # estimator = KerasRegressor(build_fn=define_model, nb_epoch=100, batch_size=5, verbose=0)
    #
    # kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=define_model, nb_epoch=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)

    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


if __name__ == "__main__":
    main()
