import numpy
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
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
    dataframe = read_csv("../data/sonar.csv", header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:60].astype(float)
    Y = dataset[:, 60]
    return X, Y


def encode(Y):
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    return encoder.transform(Y)


def define_model():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
    model.add(Dense(30, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    set_context()
    X, Y = load_data()
    encoded_Y = encode(Y)

    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=define_model, nb_epoch=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


if __name__ == "__main__":
    main()
