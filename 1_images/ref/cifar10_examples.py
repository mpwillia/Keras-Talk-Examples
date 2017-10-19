
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'

from keras.datasets import cifar10


# Stuff for Simple Networks
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

# Stuff for Advanced Network
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout
from keras import regularizers


from testing_util import test_models, print_dataset_info


def main():
    training, testing = cifar10.load_data()
    
    training = prepare_data(training) 
    testing = prepare_data(testing) 
    
    print_dataset_info(training, "Training")
    print_dataset_info(testing, "Testing")
    
    models = [small_model(), 
              large_model(), 
              smarter_model(True)]

    test_models(models, training, testing, model_prefix = 'cifar10')


def small_model(overwrite = False, name = "Small", act = 'relu',):
    model = Sequential()
    model.add(Conv2D(filters = 32, 
                     kernel_size = (7,7), 
                     activation = act,
                     input_shape = (32,32,3)))
    model.add(Conv2D(filters = 64, 
                     kernel_size = (5,5), 
                     activation = act))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation = act))
    model.add(Dense(10, activation = 'softmax'))
    
    return name, model, overwrite


def large_model(overwrite = False, name = "Large", act = 'relu'):
    model = Sequential()
    model.add(Conv2D(filters = 128, 
                     kernel_size = (7,7), 
                     activation = act,
                     input_shape = (32,32,3)))
    model.add(Conv2D(filters = 128, 
                     kernel_size = (5,5), 
                     activation = act))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters = 256, 
                     kernel_size = (3,3), 
                     activation = act))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation = act))
    model.add(Dense(128, activation = act))
    model.add(Dense(10, activation = 'softmax'))
    
    return name, model, overwrite


def smarter_model(overwrite = False, name = "Smarter", act = 'relu'):
    
    dropout_p = 0.5
    l2_str = 0.0001

    model = Sequential()
    model.add(Conv2D(filters = 32, 
                     kernel_size = (7,7), 
                     kernel_regularizer = regularizers.l2(l2_str),
                     input_shape = (32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation(act))

    model.add(Conv2D(filters = 64, 
                     kernel_size = (5,5), 
                     kernel_regularizer = regularizers.l2(l2_str)
                     ))
    model.add(BatchNormalization())
    model.add(Activation(act))

    model.add(MaxPooling2D())
    model.add(Dropout(dropout_p))
    model.add(Flatten())


    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(dropout_p))


    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    return name, model, overwrite




def prepare_data(data):
    images = data[0].astype('float32') / 255
    labels = to_categorical(data[1])
    return images, labels


if __name__ == "__main__":
    main()

