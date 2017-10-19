#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from functools import partial


from util import load_model
from southpark.line_util import clean_line, clean_whitespace
from southpark.southpark_predictive import load_dataset

from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences 


MODEL_NAME = "char_pred_demo_model"

def main():
    print("Loading model...")
    model, metadata = load_model(MODEL_NAME)
    print("Finding representative samples of output labels...")
    print("\n\n")
    good_samples = most_representative_samples(model, *metadata)
    predict_user_input(model, *metadata, good_samples = good_samples)


def predict_user_input(model, vocab, classes, good_samples = None):

    quit_keywords = {'q', 'quit', 'exit', 'bye', 'goodbye'}

    print("Enter a string and the network will try and predict which Southpark character would most likely have said it.")
    print("The Possible Characters Are:")
    for char in classes:
        print("   {}".format(char))
        if good_samples is not None:
            for samples in good_samples[char]:
                print("      '{}'".format(samples))
        print()

    print("\nTo quit enter one of the following: {}".format(quit_keywords))
    print("")

    running = True
    while running:
        try:
            user_input = input("Enter a String: ") 
            user_input = clean_line(user_input) 
            
            if user_input.lower() in quit_keywords:
                running = False
            else:
                pred = predict_string(model, user_input, vocab)
                process_output(pred, classes)

        except (KeyboardInterrupt, EOFError):
            running = False
    
    print("\nQuitting...")



def process_output(pred, classes):
    print("Predicts")
    
    if len(pred.shape) == 2:
        pred = pred[0]

    for label, prob in sorted(zip(classes, pred), key = lambda x:x[1], reverse = True):
        print("  {:12s} : {:7.2%}".format(label, prob))
    print()



def most_representative_samples(model, vocab, classes, n = 3, verbose = False):
    dataset = load_dataset(p_test = None, line_threshold = 500, clean = False)

    samples = []
    for line in dataset.inputs:
        if len(line) < 100:
            samples.append(clean_whitespace(line))

        if len(samples) > 5000:
            break

    if verbose: print("Loaded {:d} items from the dataset".format(len(samples)))
    if verbose: print("Preparing input...")
    inputs = prepare_input_multiple(samples, vocab)
    if verbose: print("Gathering predictions...")
    preds = model.predict(inputs, batch_size = 256, verbose = verbose)
    if verbose: print("Computing argmax...")
    pred_idxs = np.argmax(preds, axis = 1)
    
    
    if verbose: print("Sorting by character...")
    by_char = {char : list() for char in classes}
    for sample, pred, pred_idx in zip(samples, preds, pred_idxs):
        character = classes[pred_idx]
        conf = pred[pred_idx]
        by_char[character].append((sample, conf))

    good_samples = {char : list() for char in classes}
    for character, sample_confs in by_char.items():
        sample_confs.sort(key = lambda x:x[1], reverse = True)
        for sample, conf in sample_confs[:n]:
            good_samples[character].append(sample)
    
    return good_samples


def prepare_input_multiple(lines, vocab):
    inputs = [hashing_trick(line, len(vocab), hash_function = 'md5') for line in lines]
    return np.asarray(pad_sequences(inputs, padding = 'post'))

def predict_string(model, string, vocab):
    return model.predict(prepare_input(string, vocab))

def prepare_input(line, vocab):
    return np.asarray([hashing_trick(line, len(vocab), hash_function = 'md5')])

if __name__ == "__main__":
    main()


