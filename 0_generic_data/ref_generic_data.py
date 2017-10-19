
import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

from util import dup_dataset, dataset_stats, print_dataset_info

def main():
    training, testing = boston_housing.load_data()
    
    dataset_stats(training, testing)
    
    training = dup_dataset(training, n = 100)
    
    print_dataset_info(training, testing)
    
    act = 'sigmoid'

    # Create our network
    model = Sequential()
    model.add(Dense(256, activation = act, input_dim = 13))
    model.add(Dense(256, activation = act))
    model.add(Dense(1, activation=None))
    model.compile(optimizer = 'adam',
                  loss = 'mean_squared_error')
    
    # Train the model
    model.fit(*training, epochs = 5, batch_size = 32, verbose = 1)
    score = model.evaluate(*testing, batch_size = 32, verbose = 0)

    print("Final Score : {}".format(score))



if __name__ == "__main__":
    main()


