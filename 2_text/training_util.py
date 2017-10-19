
import random
import numpy as np
import math

from pprint import pprint

from keras.models import Sequential
from keras.layers import Embedding, Merge
from keras.layers.core import Reshape as KerasReshape
from keras import backend as K
from keras.layers.merge import Concatenate

from keras.preprocessing.sequence import pad_sequences 
from keras.utils import to_categorical 

from copy import deepcopy 


def gpu_multi_batch(model, training_data, testing_data, charset, batch_size, epochs, num_gpu_batches = 10):
    """
    
    Arguments


    """
    compile_kwargs = _extract_compile_kwargs(model)
    print("Compile Kwargs")
    pprint(compile_kwargs)

    
    chunk_map, training_batch_gen, testing_batch_gen = create_chunk_map(training_data, testing_data, charset, batch_size, num_gpu_batches)

    print_layers("Given Model", model)
    
    print_layers("Chunk Map", chunk_map)

    training_model = Sequential(chunk_map.layers + model.layers)
    
    #print_layers("Training Model", training_model)
    

    training_model.compile(**compile_kwargs)
    training_model.fit_generator(training_batch_gen, len(training_data[1]) // batch_size, 
                                 validation_data = testing_batch_gen,
                                 validation_steps = len(testing_data[1]) // batch_size,
                                 epochs = epochs,
                                 verbose = 1)
    
    return model

def print_layers(name, model):
    print("{} Layers".format(name))
    fmt = "  Layer {:2d} ({:12s}) : {:20s}  ==>  {:20s}"
    for num, layer in enumerate(model.layers):
        print(fmt.format(num, layer.name, str(layer.input_shape), str(layer.output_shape)))
    print("\n")


def create_chunk_map(training_data, testing_data, charset, batch_size, num_gpu_batches):
    chunk_size = min(num_gpu_batches * batch_size, max(len(training_data[0]), len(testing_data[0])))
    batches_per_chunk = int(math.ceil(chunk_size / batch_size))

    longest_sequence = max(len(seq) for seq in training_data[0] + testing_data[0])
    total_items = chunk_size * longest_sequence
    mb_est = (total_items * 4) / (1024*1024)

    print("Placing {:d} input sequences on GPU at a time".format(chunk_size))
    print("Full embedding size is {}x{} = {} items. Roughly {:.3} mb of data".format(chunk_size, longest_sequence, total_items, mb_est))

    def indexer_init(shape, dtype=None):
        batches_per_chunk, batch_size = shape
        
        print("Batches per Chunk : {}")
        print("Batch Size        : {}")
        
        indexer_values = []
        for idx in range(batches_per_chunk):
            indexer_values.append(np.arange(idx * batch_size, (idx+1)*batch_size))

        return np.asarray(indexer_values)

    batch_indexer = Embedding(batches_per_chunk, batch_size, 
                              trainable = False, 
                              input_length = 1,
                              embeddings_initializer = indexer_init)
    
    chunk_map = Embedding(chunk_size, longest_sequence, trainable = False, input_length = 1)

   
    # (None, 1, 64) ==> 
    chunk_mapping = Sequential()
    chunk_mapping.add(chunk_map)
    chunk_mapping.add(Reshape((longest_sequence,), 
                              input_shape = (1, longest_sequence)))
    
    training_chunk_gen = gpu_chunk_generator(chunk_map, training_data, charset, batch_size, chunk_size)
    testing_chunk_gen = gpu_chunk_generator(chunk_map, testing_data, charset, batch_size, chunk_size)

    return chunk_mapping, training_chunk_gen, testing_chunk_gen


def gpu_chunk_generator(input_chunk_map, dataset, charset, batch_size, chunk_size):
    #chunk_size = min(num_gpu_batches * batch_size, len(dataset[0]))
    longest_sequence = max(len(seq) for seq in dataset[0])
    dataset = list(zip(*dataset))
    while True:
        #print("\nLoading next GPU chunk")
        # grab multiple chunks worth of data to put on GPU
        try:
            x,y = tuple(zip(*random.sample(dataset, min(chunk_size, len(dataset)))))
        except ValueError:
            print("Dataset Size : {}".format(len(dataset)))
            print("Check Size   : {}".format(chunk_size))
            raise
        
        # setup all of our chunk data
        chunk_data = np.asarray(pad_sequences(x, maxlen = longest_sequence, padding = 'post'))
        chunk_data = np.expand_dims(chunk_data, axis = 0)
        chunk_indicies = np.asarray(list(range(len(x))))
        chunk_outputs = np.asarray(to_categorical(y, len(charset)))
        
        extra_size = chunk_size - len(x)
        if extra_size > 0:
            pad_data = np.zeros((extra_size, longest_sequence))
            pad_data = np.expand_dims(pad_data, axis = 0)
            chunk_data = np.concatenate((chunk_data, pad_data), axis = 1)

        # send the chunk data to the gpu via the embedding
        #print("Current Weights Shape : {}".format(np.asarray(chunk_map.get_weights()).shape))
        #print("Setting Weights Shape : {}".format(chunk_data.shape))
        input_chunk_map.set_weights(chunk_data)

        # iterate over the chunk indicies for the mini-batches 
        for num, minibatch in enumerate(batch((chunk_indicies, chunk_outputs), batch_size)):
            yield minibatch



                  
def _extract_compile_kwargs(model):
    
    # first detect if it has been compiled
    if not model.built: 
        raise ValueError("Model hasn't been built yet!")
    elif model.optimizer is None:
        raise ValueError("Model hasn't been compiled yet!")
    
    compile_kwargs = {'optimizer' : model.optimizer,
                      'loss' : model.loss,
                      'metrics' : model.metrics,
                      'sample_weight_mode' : model.sample_weight_mode}

    compile_kwargs.update(model.model._function_kwargs)
    
    return compile_kwargs




def batch_sample_generator(dataset, charset, batch_size = 128):
    dataset = list(zip(*dataset))
    while True:
        try:
            x,y = tuple(zip(*random.sample(dataset, batch_size)))
        except ValueError:
            print("Dataset Size : {}".format(len(dataset)))
            print("Batch Size   : {}".format(batch_size))
            raise
        x = np.asarray(pad_sequences(x, padding = 'post'))
        y = np.asarray(to_categorical(y, len(charset)))

        yield x,y


def batch_generator(dataset, charset, batch_size = 128, shuffle = True):
    while True:
        if shuffle:
            dataset = list(zip(*dataset))
            random.shuffle(dataset)
            dataset = tuple(zip(*dataset))
        
        for x,y in batch(dataset, batch_size):
            x = np.asarray(pad_sequences(x, padding = 'post'))
            y = np.asarray(to_categorical(y, len(charset)))

            yield x,y


def batch(dataset, batch_size):
    for idx in range(0,len(dataset[1]), batch_size):
        x = dataset[0][idx:idx+batch_size]
        y = dataset[1][idx:idx+batch_size]
        yield x,y




class Reshape(KerasReshape):
    def __init__(self, *args, include_batch = True, **kwargs):
        super(Reshape, self).__init__(*args, **kwargs)
        self.include_batch = include_batch

    def call(self, inputs):
        # In case the target shape is not fully defined,
        # we need access to the shape of `inputs`.
        # solution: rely on `K.int_shape`.
        target_shape = self.target_shape
        if -1 in target_shape:
            # Target shape not fully defined.
            input_shape = None
            try:
                input_shape = K.int_shape(inputs)
            except TypeError:
                pass
            if input_shape is not None:
                target_shape = self.compute_output_shape(input_shape)[1:]
        if self.include_batch:
            return K.reshape(inputs, (-1,) + target_shape)
        else:
            return K.reshape(inputs, target_shape)






