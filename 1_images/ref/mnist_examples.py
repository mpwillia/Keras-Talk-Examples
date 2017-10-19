
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'

from keras.datasets import mnist

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
    training, testing = mnist.load_data()
    training = prepare_data(training) 
    testing = prepare_data(testing) 
    
    print_dataset_info(training, "Training")
    print_dataset_info(testing, "Testing")

    #models = [small_model(), large_model(), adv_model()]
    models = [small_model(), large_model()]
    #models = [large_model(), adv_model()]
    
    compile_kwargs = {'optimizer' : 'adam',
                      'loss' : 'categorical_crossentropy',
                      'metrics' : ['accuracy']}

    test_models(models, compile_kwargs, training, testing)



def small_model(name = "Small", act = 'relu'):
    model = Sequential()
    model.add(Conv2D(filters = 32, 
                     kernel_size = (5,5), 
                     activation = act,
                     input_shape = (28,28,1)))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    
    return name, model



def large_model(name = "Large", act = 'relu'):
    model = Sequential()
    model.add(Conv2D(filters = 32, 
                     kernel_size = (7,7), 
                     activation = act,
                     input_shape = (28,28,1)))
    model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = act))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation = act))
    model.add(Dense(10, activation = 'softmax'))

    return name, model



def adv_model(name = "Advanced", act = 'relu'):
    
    conv_dropout = 0.0
    dense_dropout = 0.0
    l2_str = 0.0000

    model = Sequential()
    model.add(Conv2D(filters = 32, 
                     kernel_size = (7,7), 
                     kernel_regularizer = regularizers.l2(l2_str),
                     input_shape = (28,28,1)
                     ))
    model.add(BatchNormalization())
    model.add(Activation(act))


    model.add(Conv2D(filters = 64, 
                     kernel_size = (5,5),
                     kernel_regularizer = regularizers.l2(l2_str)
                     ))
    model.add(BatchNormalization())
    model.add(Activation(act))


    model.add(MaxPooling2D())
    model.add(Dropout(conv_dropout))
    model.add(Flatten())

    model.add(Dense(100, kernel_regularizer = regularizers.l2(l2_str)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(dense_dropout))


    model.add(Dense(10, kernel_regularizer = regularizers.l2(l2_str)))   
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return name, model



def prepare_data(data):
    images = data[0].astype('float32') / 255
    labels = to_categorical(data[1])
    return images.reshape(images.shape[0], 28, 28, 1), labels


if __name__ == "__main__":
    main()

