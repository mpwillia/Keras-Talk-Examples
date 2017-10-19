
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from util import save_model, load_model
from dataset import load, load_all, load_artists

artists = ["Taylor Swift", "Kanye West", "Bonobo", "Eminem", "Green Day", 
           "Beastie Boys", "Rammstein", "John Lennon", "The Beatles", 
           "David Bowie", "Emancipator", "Of Monsters And Men", 
           "Beats Antique", "Childish Gambino"]

def main():

   dataset, classes = load_artists(artists)
   
   print("\n")
   print_dataset_shape(dataset)
  
   dataset = prepare_dataset(dataset, classes)

   from keras.callbacks import ModelCheckpoint
   model = create_model(dataset[0].shape, classes)
  
   filepath="artist_predictor_best.h5"
   checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
   #callbacks_list = [checkpoint]
   callbacks_list = None
   model.fit(*dataset, batch_size = 16, epochs = 30, verbose = 1, validation_split = 0.05, callbacks = callbacks_list)

   #save_model("artist_predictor_final", model, classes, overwrite = True)


def create_model(input_shape, classes):
   from keras.models import Sequential
   from keras.layers import Dense, Activation, Flatten, Reshape, Dropout
   from keras.layers.recurrent import GRU
   from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D
   from keras.layers.normalization import BatchNormalization
   from keras.regularizers import l2
   from keras import optimizers

   freq_axis = 1
   time_axis = 2
   channel_axis = 3
   
   l2_str = 0.0
   dropout_p = 0.1

   model = Sequential()

   # Input Block
   model.add(ZeroPadding2D(padding = (1, 37), input_shape = input_shape[1:]))
   model.add(BatchNormalization(axis = time_axis))
   
   # Conv Block 1
   model.add(Conv2D(64, (3, 3), padding = "same", kernel_regularizer = l2(l2_str)))
   model.add(BatchNormalization(axis = channel_axis))
   model.add(Activation('elu'))
   model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
   model.add(Dropout(dropout_p))

   # Conv Block 2
   model.add(Conv2D(128, (3, 3), padding = "same", kernel_regularizer = l2(l2_str)))
   model.add(BatchNormalization(axis = channel_axis))
   model.add(Activation('elu'))
   model.add(MaxPooling2D(pool_size = (3,3), strides = (3,3)))
   model.add(Dropout(dropout_p))

   # Conv Block 3
   model.add(Conv2D(128, (3, 3), padding = "same", kernel_regularizer = l2(l2_str)))
   model.add(BatchNormalization(axis = channel_axis))
   model.add(Activation('elu'))
   model.add(MaxPooling2D(pool_size = (4,4), strides = (4,4)))
   model.add(Dropout(dropout_p))

   # Conv Block 4
   model.add(Conv2D(128, (3, 3), padding = "same", kernel_regularizer = l2(l2_str)))
   model.add(BatchNormalization(axis = channel_axis))
   model.add(Activation('elu'))
   model.add(MaxPooling2D(pool_size = (4,4), strides = (4,4)))
   model.add(Dropout(dropout_p))

   # Reshaping
   conv_end_shape = model.layers[-1].output_shape
   model.add(Reshape(conv_end_shape[2:], input_shape = conv_end_shape))

   # GRU blocks 
   model.add(GRU(32, return_sequences = True, kernel_regularizer = l2(l2_str)))
   model.add(GRU(32, return_sequences = False, kernel_regularizer = l2(l2_str)))
   model.add(Dropout(0.3))

   # GRU blocks 
   model.add(Dense(len(classes), activation = 'softmax'))
   
   print_layers("Final Model", model)

   model.compile(loss = 'categorical_crossentropy',
                 optimizer = optimizers.Nadam(lr = 0.0003),
                 metrics = ['accuracy'])
   
   return model



def print_layers(name, model):
    print("{} Layers".format(name))
    fmt = "  Layer {:2d} ({:24s}) : {:20s}  ==>  {:20s}"
    for num, layer in enumerate(model.layers):
        print(fmt.format(num, layer.name, str(layer.input_shape), str(layer.output_shape)))
    print("\n")


def print_last_layer(model):
   layer = model.layers[-1]
   fmt = "  Layer {:2d} ({:24s}) : {:20s}  ==>  {:20s}"
   print(fmt.format(len(model.layers)-1, layer.name, str(layer.input_shape), str(layer.output_shape)))



def prepare_dataset(dataset, classes):
   from keras.utils import to_categorical 
   inputs = dataset[0] 
   labels = np.asarray(to_categorical(dataset[1], len(classes)))
   return (inputs, labels)



def print_dataset_shape(dataset):
   print("Dataset Shape : {}  [(batch, freq, time, channel)]".format(dataset[0].shape))


if __name__ == "__main__":
    main()


