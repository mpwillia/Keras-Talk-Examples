
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

from testing_util import print_dataset_info

def main():
    training, testing = mnist.load_data()
    training = prepare_data(training)
    testing = prepare_data(testing)

    print_dataset_info(training, "Training")
    print_dataset_info(testing, "Testing")
    
    
    model = Sequential()
    model.add(Conv2D(filters = 64,
                     kernel_size = (5,5),
                     activation = 'relu',
                     input_shape = (28,28,1)))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'adam', 
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    model.fit(*training, epochs = 5, batch_size = 128, verbose = 1, validation_split = 0.2)

    cost, acc = model.evaluate(*testing, batch_size = 128, verbose = 0)


    print("Results")
    print("  Cost     : {:7.3f}".format(cost))
    print("  Accuracy : {:7.2%}".format(acc))
    print()


def prepare_data(data):
    images = data[0].astype('float32') / 255
    labels = to_categorical(data[1])
    return images.reshape(images.shape[0], 28, 28, 1), labels



if __name__ == "__main__":
    main()

