'''
The problem that we will look at in this demo is the Boston house price dataset.
The dataset describes properties of houses in Boston suburbs and is concerned with modeling the price of houses in those suburbs in thousands of dollars.
As such, this is a regression predictive modeling problem.

There are 13 input variables that describe the properties of a given Boston suburb.
The full list of attributes in this dataset are as follows:
1. CRIM: per capita crime rate by town.
2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS: proportion of non-retail business acres per town.
4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5. NOX: nitric oxides concentration (parts per 10 million).
6. RM: average number of rooms per dwelling.
7. AGE: proportion of owner-occupied units built prior to 1940.
8. DIS: weighted distances to five Boston employment centers.
9. RAD: index of accessibility to radial highways.
10. TAX: full-value property-tax rate per 10,000.
11. PTRATIO: pupil-teacher ratio by town.
12. B: 1000(Bk   0.63)2 where Bk is the proportion of blacks by town.
13. LSTAT: % lower status of the population.

14. MEDV: Median value of owner-occupied homes in 1000s.
'''

import numpy
from keras.utils.visualize_util import plot
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data():
    dataframe = read_csv("../data/housing.csv", delim_whitespace=True, header=None)
    dataset = dataframe.values
    attributes = dataset[:, 0:13]
    prices = dataset[:, 13]
    return attributes, prices


def create_a_neural_network():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    # Compile model
    # definition of MSE: https://en.wikipedia.org/wiki/Mean_squared_error
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # show the basic neural network structure
    plot(model, to_file='./NeuralNetworkToPredictHousePrice.png', show_shapes=True)

    return model


def main():
    # fix random seed for reproducibility
    seed = 123
    numpy.random.seed(seed)

    attributes, prices = load_data()

    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=create_a_neural_network, nb_epoch=500, batch_size=5, verbose=2)))
    pipeline = Pipeline(estimators)

    # k-fold cross validation
    kfold = KFold(n_splits=5, random_state=seed)
    results = cross_val_score(pipeline, attributes, prices, cv=kfold)

    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


if __name__ == "__main__":
    main()
