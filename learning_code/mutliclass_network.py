import numpy
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix the random seed
seed = 7


def set_context():
    # fix random seed for reproducibility
    numpy.random.seed(seed)


def load_data():
    # load dataset
    dataframe = read_csv("../data/iris.csv", header=None)
    dataset = dataframe.values
    X = dataset[:, 0:4].astype(float)
    Y = dataset[:, 4]
    return X, Y


def encode(Y):
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y


def define_model():
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    set_context()
    X, Y = load_data()
    dummy_y = encode(Y)

    estimator = KerasClassifier(build_fn=define_model, nb_epoch=200, batch_size=5, verbose=0)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


if __name__ == "__main__":
    main()
