#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import sys

from util import load_model
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences 

from southpark.southpark_generative import string_one_hot, char_one_hot


MODEL_NAME = "script_gen_demo_model"

def main():
    print("Loading model...")    
    model, charset = load_model(MODEL_NAME)
    
    print(charset)

    seed_text = input("Enter a String: ").strip()
    print()
    generate_script(seed_text, model, charset)

def generate_script(seed_text, model, charset):
    
    sys.stdout.write(seed_text)
    sys.stdout.flush()
    next_char = None
    should_stop = False
    while not should_stop:
        prev_char = next_char
        next_char = sample(model, seed_text, charset, temp = 0.2)
        
        sys.stdout.write(next_char)
        sys.stdout.flush()
        
        if prev_char == '\n' and prev_char == next_char:
            should_stop = True

    
def sample(model, string, charset, temp = 1.0):
    inputs = [string_one_hot(string, charset)]
    inputs = pad_sequences(inputs, padding = 'post', maxlen = 64)
    preds = model.predict(inputs)[0]
    
    return charset[sample_preds(preds, temp)]


def sample_preds(results, temperature = 1.0):
    # helper function to sample an index from a probability array

    if temperature <= 0.0:
        return np.argmax(results)
    
    #num_choices = results.shape[0] # (batch, outputs)
    probs = np.exp(np.log(results) / temperature)
    probs /= np.sum(probs)
    return np.random.choice(len(results), p = probs)


    #preds = np.asarray(preds).astype('float64')
    #preds = np.log(preds) / temperature
    #exp_preds = np.exp(preds)
    #preds = exp_preds / np.sum(exp_preds)
    #probas = np.random.multinomial(1, preds, 1)
    #
    #print(probas)

    #return np.argmax(probas)




if __name__ == "__main__":
    main()

