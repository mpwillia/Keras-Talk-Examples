
import os
import keras
import pickle

MODEL_DIR = "./models"


def save_model(name, model, metadata, overwrite = False):
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_path, metadata_path = get_paths(name) 
    
    if os.path.exists(model_path) and not overwrite:
        raise Exception("Model already exists and told to not overwrite!")
    
    model.save(model_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


def load_model(name):
    
    model_path, metadata_path = get_paths(name) 

    model = keras.models.load_model(model_path)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, metadata


def get_paths(name):

    model_name = name + '.h5'
    metadata_name = name + '-metadata.pkl'

    model_path = os.path.join(MODEL_DIR, model_name)
    metadata_path = os.path.join(MODEL_DIR, metadata_name)
    
    return model_path, metadata_path


