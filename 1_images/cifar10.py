
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'theano'

from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

from util import print_dataset_info


def main():
    training, testing = cifar10.load_data()
   
    training = prepare_data(training)
    testing = prepare_data(testing)

    print_dataset_info("Training", training)
    print_dataset_info("Testing", testing)
    
    model = Sequential()
    model.add(Conv2D(128, (3,3), activation = 'relu', input_shape = (32,32,3)))
    model.add(Conv2D(128, (3,3), activation = 'relu'))

    model.add(Flatten())

    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    model.compile(loss = "categorical_crossentropy",
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    model.fit(*training, epochs = 5, batch_size = 128)
    loss, acc = model.evaluate(*testing, batch_size = 128)



    print("\n\nFinal Result")
    print("\tLoss : {:7.3f}".format(loss))
    print("\tAcc : {:7.2%}".format(acc))


def prepare_data(data):
    inputs = data[0].astype('float32') / 255.0
    outputs = to_categorical(data[1], 10)
    return inputs, outputs



if __name__ == "__main__":
    main()

