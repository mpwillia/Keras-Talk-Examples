
import numpy as np

def print_dataset_info(name, dataset):
    
    def get_info(data):
        return str(data.shape), str(data.dtype), np.amin(data), np.amax(data)
    
    info_fmt = "  {0:6s} :   type {2:6s} | shape {1:18s} | range [ {3:.3f} , {4:.3f} ]"

    print("{} Dataset".format(name))
    print(info_fmt.format("Input", *get_info(dataset[0])))
    print(info_fmt.format("Output", *get_info(dataset[1])))
    print()


