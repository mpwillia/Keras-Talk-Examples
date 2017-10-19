
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'theano'

import southpark
import random
import numpy as np

from keras.preprocessing.sequence import pad_sequences 
from keras.utils import to_categorical 

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from training_util import batch_sample_generator, batch_generator, gpu_multi_batch
from util import save_model

def main():
    #filter_seasons = set([10])
    filter_seasons = None
    min_size = 56
    max_size = 64
    training, testing, charset = southpark.load_generative_data(min_size = min_size, max_size = max_size, filter_seasons = filter_seasons, dataset_size = 500000)
    
    print("Dataset")
    print("  Training Size : {}".format(len(training[1])))
    print("  Testing Size  : {}".format(len(testing[1])))
    print("  Charset Size  : {}".format(len(charset)))
    print("  Charset       : {}".format(charset))
    print()
    
    print("Creating Model...")
    model = create_model(charset, max_size)
    
    batch_size = 128
    use_gpu_multi_batching = False
    
    if use_gpu_multi_batching:
        model = gpu_multi_batch(model, training, testing, charset, batch_size, 
                                        epochs = 5,
                                        num_gpu_batches = 1000)
    else:
        #batch_gen = batch_sample_generator(dataset, charset, batch_size)
        batch_gen = batch_generator(training, charset, batch_size)

        print("Fitting Model...")

        filepath="script_gen_best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit_generator(batch_gen, len(training[1]) // batch_size, 
                            epochs = 20,
                            use_multiprocessing = False,
                            callbacks = callbacks_list,
                            verbose = 1)

    save_model("script_gen_20_epoch", model, charset, overwrite = True)


def create_model(charset, max_size):
    dropout = 0.0
    l2_str = 0.00001
    
    reg = lambda:regularizers.l2(l2_str)
    node_type = GRU
    lstm_kwargs = {#'dropout' : dropout,
                   #'recurrent_dropout' : dropout,
                   }

    model = Sequential()
    model.add(Embedding(len(charset), len(charset), embeddings_regularizer = reg(), mask_zero = True))
    model.add(node_type(256, kernel_regularizer = reg(), return_sequences = True, **lstm_kwargs))
    #model.add(node_type(256, kernel_regularizer = reg(), return_sequences = True, **lstm_kwargs))
    model.add(node_type(256, kernel_regularizer = reg(), **lstm_kwargs))

    #model.add(Dense(256, kernel_regularizer = reg()))
    #model.add(BatchNormalization())
    #model.add(Activation('tanh'))

    model.add(Dense(len(charset), kernel_regularizer = reg()))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizers.Nadam(lr=0.001),
                  metrics = ['accuracy'])

    return model



        


if __name__ == "__main__":
    main()

