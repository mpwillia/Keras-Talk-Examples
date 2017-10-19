
import os
import csv
import numpy as np
import random

from collections import defaultdict
import itertools

from .line_util import is_ascii

#from keras.preprocessing.sequence import pad_sequences 
#from keras.utils import to_categorical 

MODULE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(MODULE_DIR,"all-seasons.csv")

def load_generative_data(path = CSV_PATH, filter_seasons = None, min_size = None, max_size = 128, p_test = 0.1, dataset_size = None):
    
    print("Loading Scripts...")
    episode_scripts = load_dataset(path, filter_seasons)

    kwargs = {'min_size' : min_size,
              'max_size' : max_size,
              'p_test' : p_test,
              'dataset_size' : dataset_size}

    #dataset, charset = make_dataset(episode_scripts, **kwargs)

    print("Creating Dataset...")
    return make_dataset(episode_scripts, **kwargs)


def make_dataset(episode_scripts, dataset_size = None, p_test = 0.1,  **kwargs):
    charset = get_charset(episode_scripts)
    scripts = [string_one_hot(v, charset) for v in episode_scripts.values()]
    
    dataset = []
    
    inputs = []
    outputs = []

    for script in scripts:
        for input_seq, output_char in sliding_window(script, **kwargs):
            inputs.append(input_seq)
            outputs.append(output_char)
    
    dataset = list(zip(inputs, outputs))
    random.shuffle(dataset)
    
    if dataset_size is not None and dataset_size > 0:
        dataset = dataset[:dataset_size]

    if p_test is not None and p_test > 0:
        testing = dataset[:int(len(dataset)*p_test)]
        training = dataset[int(len(dataset)*p_test):]

        testing = tuple(zip(*testing))
        training = tuple(zip(*training))
        
        #inputs = np.asarray(pad_sequences(inputs, padding='post'))
        #outputs = np.asarray(to_categorical(outputs, len(charset)))

        return training, testing, charset
    else:
        training = tuple(zip(*training))
        return training, charset

def sliding_window(script, min_size = None, max_size = None):
    if min_size is None or min_size <= 0:
        min_size = 1
    
    for idx in range(min_size, len(script)):
        output_char = script[idx]   
    
        if max_size is not None:
            start_idx = max(idx - max_size, 0)
            input_seq = script[start_idx:idx]
        else:
            input_seq = script[:idx]
        
        yield input_seq, output_char


def make_dataset_one_hot(episode_scripts, char_set_size):
    return {k:string_one_hot(v, char_set_size) for k,v in episode_scripts.items()}

def string_one_hot(string, charset):
    return np.asarray([charset.index(c) for c in string])

def char_one_hot(char, charset):
    return charset.index(char)

def get_charset(episode_scripts):
    charset = set()
    for script in episode_scripts.values():
        charset.update(set(script))
    return sorted(list(charset))


def load_dataset(path, filter_seasons = None):
    
    episode_lines = defaultdict(list)
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            try:
                season = int(row['Season'])
                episode = int(row['Episode'])
            except ValueError:
                continue

            char = row['Character']
            line = row['Line'].strip()

            if is_ascii(line):
                char_line = "{}: {}".format(char.upper(), line)
                char_line = char_line.replace('*', '')
                char_line = char_line.replace('É', 'E')
                char_line = char_line.replace('Ñ', 'N')
                 
                if not is_ascii(char_line):
                    print(char_line)

                episode_lines[(season, episode)].append(char_line)
    
    episode_scripts = dict()
    for which, lines in episode_lines.items():
        season, episode = which
        if filter_seasons is not None and season not in filter_seasons:
            continue

        script = '\n'.join(lines) + "\n"
        episode_scripts[which] = script
    
    return episode_scripts



