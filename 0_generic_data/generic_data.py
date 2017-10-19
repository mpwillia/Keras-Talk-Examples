
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np


from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

from util import print_dataset_info, dup_dataset, dataset_stats


def main():
    training, testing = boston_housing.load_data()

    #training = dup_dataset(training, 100)

    dataset_stats(training, testing)
    print_dataset_info(training, testing)

    model = Sequential()
    model.add(Dense(128, activation = 'sigmoid', input_dim = 13))
    model.add(Dense(128, activation = 'sigmoid'))
    model.add(Dense(1, activation = None))

    model.compile(loss = 'mean_squared_error',
                  optimizer = 'adam')

    model.fit(training[0], training[1], epochs = 500, batch_size = 32)
    loss = model.evaluate(*testing, batch_size = 32)

    print("\n\nFinal Loss : {}".format(loss))

if __name__ == "__main__":
    main()

