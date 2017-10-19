
import os
import csv
import random
from collections import defaultdict
import numpy as np

from .line_util import clean_line
from . import dataset_util
from .dataset_util import Dataset
from .dataset_util import apply_word_count_threshold, compute_vocab_with_stats

from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick
from keras.preprocessing.sequence import pad_sequences 

MODULE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(MODULE_DIR,"all-seasons.csv")

def load_predictive_data(path = CSV_PATH, test_split = 0.1, char_line_threshold = 1000,
    line_min_words = 3, line_max_words = None, 
                         vocab_min_freq = 5,
                         vocab_max_freq = None,
                         vocab_min_tfidf = 0.5,
                         vocab_max_tfidf = None,
                         replacement_token = "<pruned_word>"
                         ):
   
    training, testing = load_dataset(path, test_split, char_line_threshold)
   
   # word count prune
    wc_kwargs = {'min_words' : line_min_words, 'max_words' : line_max_words}
    training = apply_word_count_threshold(training, **wc_kwargs)
    testing = apply_word_count_threshold(testing, **wc_kwargs)

    vocab = compute_vocab_with_stats(training.inputs, testing.inputs)
   
   # tfidf prune
    tfidf_kwargs = {'vocab' : vocab, 
                    'min_thresh' : vocab_min_tfidf, 
                    'max_thresh' : vocab_max_tfidf,
                    'replacement' : replacement_token}
    training = dataset_util.trim_dataset_by_tfidf(training, **tfidf_kwargs)
    testing = dataset_util.trim_dataset_by_tfidf(testing, **tfidf_kwargs)
   
    vocab = compute_vocab_with_stats(training.inputs, testing.inputs)
    vocab = dataset_util.trim_vocab_by_freq(vocab, vocab_min_freq, vocab_max_freq)
   
   # word freq prune
    vocab_kwargs = {'vocab' : vocab, 
                    'replacement' : replacement_token}
    training = dataset_util.trim_dataset_by_vocab(training, **vocab_kwargs)
    testing = dataset_util.trim_dataset_by_vocab(testing, **vocab_kwargs)

   # final preperations
    vocab = compute_vocab_with_stats(training.inputs, testing.inputs)
    classes = sorted(list(set(training.labels + testing.labels)))

    training = prepare_data(training, classes, vocab)
    testing = prepare_data(testing, classes, vocab)

    return training, testing, vocab, classes


def prepare_data(data, classes, vocab):
    inputs = []
    labels = []
    for line, char in zip(*data):
        #inputs.append(one_hot(line, len(vocab)))
        inputs.append(hashing_trick(line, len(vocab), hash_function = 'md5'))

        one_hot_out = np.zeros(len(classes))
        one_hot_out[classes.index(char)] = 1
        labels.append(one_hot_out)
    
    return np.asarray(pad_sequences(inputs, padding='post')), np.asarray(labels)


def load_dataset(path = CSV_PATH, p_test = 0.1, line_threshold = 1000, shuffle = True, clean = True):
    lines_by_char = defaultdict(list)
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            lines_by_char[row['Character']].append(row['Line'])
    
    data = [] 
    for char, lines in lines_by_char.items():
        if len(lines) >= line_threshold:
            for line in lines:
                if clean:
                    data.append((clean_line(line), char))
                else:
                    data.append((line, char))
   
    if shuffle:
        random.shuffle(data)
   

    def split(data):
        """Converts list of pairs into a pair of lists"""
        return Dataset(*tuple(zip(*data)))

    if p_test is not None and p_test > 0:
        testing = data[:int(len(data)*p_test)]
        training = data[int(len(data)*p_test):]
        return split(training), split(testing)
    else:
        return split(data)


