
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'theano'

import csv
import random
from collections import defaultdict, namedtuple
import numpy as np
import string
import itertools

from keras.preprocessing.text import text_to_word_sequence, one_hot
from keras.preprocessing.sequence import pad_sequences 

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import optimizers

import southpark
from util import save_model


def main():
    kwargs = {'char_line_threshold' : 1000,
              'line_min_words' : 3,
              'line_max_words' : 64,
              'vocab_min_freq' : 5,
              'vocab_min_tfidf': 0.75}
    
    training, testing, vocab, classes = southpark.load_predictive_data(**kwargs)
    print("Vocab Size    : {}".format(len(vocab)))
    print("Output Labels : {}".format(classes))
    print()
    print("Training Dataset")
    print("  Inputs Shape : {}".format(training[0].shape))
    print("  Labels Shape : {}".format(training[1].shape))
    print()
    print("Testing Dataset")
    print("  Inputs Shape : {}".format(testing[0].shape))
    print("  Labels Shape : {}".format(testing[1].shape))
    print

    model = create_model(vocab, classes) 

    model.fit(*training, 
              epochs = 100, 
              batch_size = 128, 
              validation_data = testing)
    
    save_model("char_pred_model_100_epoch", 
               model, 
               ({k:tuple(v) for k,v in vocab.items()}, classes), 
               overwrite = True)

    score, acc = model.evaluate(*testing, batch_size = 128)

    print("\nResults")
    print("  Cost     : {:7.3f}".format(score))
    print("  Accuracy : {:7.2%}".format(acc))
    print()
    


def create_model(vocab, classes):
    dropout = 0.5
    l2_str = 0.0001
    
    lstm_kwargs = {'dropout' : dropout,
                   'recurrent_dropout' : dropout,
                   }

    model = Sequential()
    # Setup Vocabulary Embedding
    model.add(Embedding(len(vocab), 128, embeddings_regularizer = l2(l2_str), mask_zero = True))

    # Setup Our 3 LSTM Blocks
    model.add(LSTM(256, kernel_regularizer = l2(l2_str), return_sequences = True, **lstm_kwargs))
    model.add(LSTM(256, kernel_regularizer = l2(l2_str), return_sequences = True, **lstm_kwargs))
    model.add(LSTM(128, kernel_regularizer = l2(l2_str), **lstm_kwargs))
    
    # Dense Layer
    model.add(Dense(128, kernel_regularizer = l2(l2_str)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(dropout))
    
    # Output Layer
    model.add(Dense(len(classes), kernel_regularizer = l2(l2_str)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))


    model.compile(loss = 'categorical_crossentropy',
                  optimizer = optimizers.Nadam(lr=0.0003),
                  metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    main()

