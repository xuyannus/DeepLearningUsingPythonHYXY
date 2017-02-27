from keras.models import Sequential
from keras.layers import Dense
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


def define_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def calibrate_model(X, Y, model):
    # Fit the model
    model.fit(X, Y, nb_epoch=150, batch_size=10, validation_split=0.33)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def main():
    set_context()
    X, Y = load_data()
    model = define_model()
    calibrate_model(X, Y, model)


if __name__ == "__main__":
    main()
