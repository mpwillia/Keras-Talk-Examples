#!/home/mike/anaconda3/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import subprocess
import sys
import glob

from util import load_model
from audio_util import preprocess_input 

DOWNLOAD_DST = "/mnt/Data/Development/artist_classifier/youtube_downloads"
MODEL_NAME = "artist_predictor_demo_model"

def main():
    model, classes = load_model(MODEL_NAME)
    prompt_for_youtube_id(model, classes)


def prompt_for_youtube_id(model, classes):
    
    quit_keywords = {'q', 'quit', 'exit', 'bye', 'goodbye'}
    
    running = True
    while running:
        try:
            print("\n")
            user_input = input("Enter a Youtube URL or Video ID: ") 
        
            if user_input in quit_keywords:
                running = False
            else:  
                video_file = download_youtube_video(user_input, verbose = False)

                if video_file is not None:
                    print("Preprocessing input file...")
                    input_data = np.expand_dims(preprocess_input(video_file), axis = 0)
                    preds = model.predict(input_data)[0]
                    print_preds(preds, classes, os.path.basename(video_file))
            
        except (KeyboardInterrupt, EOFError):
            running = False
    
    print("\nQuitting...")


def print_preds(preds, classes, filepath = None):
    fmt = "  {:" + str(max(len(c) for c in classes)) + "s} : {:7.2%}"
    print()
    if filepath is not None:
        print("'{}' is most like:".format(extract_video_name(filepath)))
    for pred, label in sorted(zip(preds, classes), key = lambda x:x[0], reverse = True):
        print(fmt.format(label, pred))

def extract_video_name(filepath):
    return os.path.basename(filepath).rsplit("[", 1)[0]
    

def download_youtube_video(url, output_dir = DOWNLOAD_DST, verbose = False):
    
    if "watch?v=" not in url:
        url = "www.youtube.com/watch?v=" + url
    

    ytid = url.split("?v=")[1]
    if "&" in ytid:
        ytid = ytid.split("&")[0]
    
    pattern = "*" + glob.escape("[" + ytid + "]") + "*"
    check_files = glob.glob(os.path.join(output_dir, pattern))
    if len(check_files) > 0:
        print("File already downloaded")
        return check_files[0]
    
    print("Downloading video...")
    command = ['youtube-dl', '-o', os.path.join(output_dir, "%(title)s[%(id)s].%(ext)s"), "-x", "--audio-format", "mp3", url]
    proc = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    
    if verbose:
        for line in iter(proc.stdout.readline, b''):
            print(line.rstrip().decode('utf-8'))
    
    out, err = proc.communicate()

    check_files = glob.glob(os.path.join(output_dir, pattern))
    if len(check_files) > 0:
        print("Successfully downloaded video")
        return check_files[0]
    else:
        print("There was an error downloading the video, try a different one!")
        return None


if __name__ == "__main__":
    main()

