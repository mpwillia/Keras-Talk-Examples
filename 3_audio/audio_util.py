
import numpy as np
import time

import librosa
from librosa import power_to_db
from librosa.feature import melspectrogram
from librosa.display import specshow

def preprocess_input(audio_path, 
                     add_channel_dim = True, 
                     sample_rate = 12000, 
                     n_fft = 512, 
                     hop_len = 256, 
                     n_mels = 96, 
                     duration = 29.12, 
                     verbose = False):
    '''Reads an audio file and outputs a Mel-spectrogram.
    '''

    # mel-spectrogram parameters
    sample_rate = sample_rate
    #n_fft = 512 # length of the FFT window
    #n_mels = 96 # number of mel bands to generate
    #hop_len = 256 # number of samples between successive frames
    duration = duration
    
    if verbose:
        print("Spectrogram Args")
        print("  Sample Rate  : {:d}".format(sample_rate))
        print("  Num FFT      : {:d}".format(n_fft))
        print("  Hop Length   : {:d}".format(hop_len))
        print("  Num Mel Bins : {:d}".format(n_mels))
        print("  Duration     : {:7.3}".format(duration))
        print("")

    #n_fft = 512 # length of the FFT window
    #n_mels = 96 # number of mel bands to generate
    #hop_len = 256 # number of samples between successive frames


    src, sr = librosa.load(audio_path, sr = sample_rate)
    n_sample = src.shape[0]
    n_sample_wanted = int(duration * sample_rate)

    # trim the signal at the center
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((n_sample_wanted - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[(n_sample - n_sample_wanted) // 2:
                  (n_sample + n_sample_wanted) // 2]

    melgram_kwargs = {'sr' : sample_rate,
                      'hop_length' : hop_len,
                      'n_fft' : n_fft,
                      'n_mels' : n_mels,
                      'power' : 2.0}

    x = melspectrogram(y = src, **melgram_kwargs)
    x = power_to_db(x, ref = 1.0)
    
    if add_channel_dim:
        # add channel dimension
        x = np.expand_dims(x, axis=3)

    return x








def main():
    import os
    TEST_FILES_DIR = "/mnt/Data/Development/artist_classifier/test_files"

    
    music_files = find_test_files(TEST_FILES_DIR)

    print("Found {:d} files for testing".format(len(music_files)))
    for n, f in enumerate(music_files):
        print("  {:2d} : '{}'".format(n, os.path.basename(f)))
    
    tswift = [f for f in music_files if "Taylor Swift" in f][0]
    emancipator = [f for f in music_files if "Emancipator" in f][0]
    kanye = [f for f in music_files if "Kanye West" in f][0]
    
    import matplotlib.pyplot as plt

    def show_spec_info(filepath):
        print("Computing Spectrogram for '{}'".format(filepath))
        #hq_db_spec = preprocess_input(filepath, False)
        #lq_db_spec = preprocess_input(filepath, False, sample_rate = 12000, n_fft = 512, n_mels = 96, hop_len = 256)

        hq_db_spec = time_func(preprocess_input, filepath, False)
        lq_db_spec = time_func(preprocess_input, filepath, False, sample_rate = 12000, n_fft = 512, n_mels = 96, hop_len = 256)

        print("Shapes")
        print("  High Quality Shape : {}  [{:d} elements]".format(hq_db_spec.shape, np.prod(hq_db_spec.shape)))
        print("  Low Quality Shape  : {}  [{:d} elements]".format(lq_db_spec.shape, np.prod(hq_db_spec.shape)))

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle("Spectrogram's for '{}'".format(os.path.basename(filepath)))
        plt.subplot(2, 1, 1)
        specshow(hq_db_spec, y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title("High Quality Mel Spectrogram")

        plt.subplot(2, 1, 2)
        specshow(lq_db_spec, y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Low Quality Mel Spectrogram")

        plt.tight_layout()
        print("")

    show_spec_info(tswift)
    show_spec_info(emancipator)
    show_spec_info(kanye)
    show_spec_info(librosa.util.example_audio_file())
    
    plt.show()

def time_func(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    print("Took {:7.3} seconds".format(time.time() - start))
    return result


def find_test_files(root):
    from dataset import find_music_files    
    return sorted(find_music_files(root, {".mp3", ".m4a", ".flac"}))


if __name__ == "__main__":
    main()



