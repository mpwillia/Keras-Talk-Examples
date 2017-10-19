
import os
import sys
from glob import glob
import numpy as np
import pickle
from multiprocessing import Pool

from audio_util import preprocess_input

#PREPROCESSED_DATA = "./preprocessed"
PREPROCESSED_DATA = "/mnt/Data/Development/artist_classifier/preprocessed"

def load(class_dirs):
    
    class_files = dict()
    for label, dir in class_dirs.items():
        class_files[label] = find_music_files(dir)
    

    for label, files in class_files.items():
        print("{} : {:d} Songs".format(label, len(files)))
   
    classes = sorted(list(class_files.keys()))

    inputs = []
    labels = []
    
    for label, files in class_files.items():
        
        label, loaded_files = load_class_files(label, files)
        label_idx = classes.index(label)
        
        for f in loaded_files:
            inputs.append(f)
            labels.append(label_idx)
    

    return (np.asarray(inputs), np.asarray(labels)), classes


def load_all(root_dir, min_files):
    
    artist_dirs = [f for f in glob(os.path.join(root_dir, "*")) if os.path.isdir(f)]
    
    class_files = dict()
    for artist_dir in artist_dirs:
        artist = os.path.basename(artist_dir)
        files = find_music_files(artist_dir)

        if len(files) > min_files and '_' not in artist:
            class_files[artist] = files
    
    classes = sorted(list(class_files.keys()))
    print("\n{:d} Classes".format(len(classes)))

    inputs = []
    labels = []
    
    for label, files in class_files.items():
        
        label, loaded_files = load_class_files(label, files)
        label_idx = classes.index(label)
        
        for f in loaded_files:
            inputs.append(f)
            labels.append(label_idx)
    

    return (np.asarray(inputs), np.asarray(labels)), classes   


def load_artists(artists):
    
    classes = sorted(list(artists))
    print("\n{:d} Classes".format(len(classes)))

    inputs = []
    labels = []

    for artist in artists:
        label, loaded_files = load_class_files(artist, None)

        label_idx = classes.index(label)
        
        for f in loaded_files:
            inputs.append(f)
            labels.append(label_idx)
    
    return (np.asarray(inputs), np.asarray(labels)), classes   


def load_class_files(label, files):
    print("Loading song data for '{}'".format(label)) 
    pkl_path = "{}-preprocessed.pkl".format(label.replace(" ", "_"))
    pkl_path = os.path.join(PREPROCESSED_DATA, pkl_path)
    
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    elif files is not None:
        print("No existing data found, creating...")
       
        if not os.path.exists(PREPROCESSED_DATA):
            os.makedirs(PREPROCESSED_DATA)

        proc_pool = Pool(processes = 8)
        loaded_files = proc_pool.map(preprocess_input, files)
        proc_pool.close()
        proc_pool.join()
        
        data = (label, loaded_files)

        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

        return data
    else:
        raise ValueError("No existing data found for '{}' and not given files to generate it from!".format(label))

def find_music_files(root, allowed_formats = {".mp3", ".m4a"}):
    music_files = []
    for root, dirs, files in os.walk(root):
        for f in files:
            if ext_is(f, allowed_formats):
                music_files.append(os.path.join(root, f))
    
    return music_files



def ext_is(path, valid_ext):
    """
    Checks if the given path ends with a file extension that either matches
    the one given or is in the set of valid extensions given.
    """
    ext = os.path.splitext(path)[1]
    if isinstance(valid_ext, str):
        return ext == valid_ext
    else:
        return ext in valid_ext
        





