
import numpy as np

def dup_dataset(dataset, n = 5):
    inputs = np.concatenate([dataset[0] for _ in range(n)]) 
    outputs = np.concatenate([dataset[1] for _ in range(n)]) 
    return inputs, outputs


def print_dataset_info(training, testing):
    print("Training Dataset")
    print("  Input Shape  : {}".format(training[0].shape))
    print("  Output Shape : {}".format(training[1].shape))
    print()
    print("Testing Dataset")
    print("  Input Shape  : {}".format(testing[0].shape))
    print("  Output Shape : {}".format(testing[1].shape))
    print()


def dataset_stats(*data):
    all_inputs = np.concatenate([d[0] for d in data])
    all_outputs = np.concatenate([d[1] for d in data])

    data = np.concatenate((all_inputs, np.expand_dims(all_outputs, axis = 1)), axis = 1)

    stats = ['Crime Rate', 
             'Zoning', 
             'Industry', 
             'Charles River', 
             'Nitric Oxides',
             'Avg Rooms', 
             'Built Prior to 1940', 
             'Employment Center Distance',
             'Highway Access', 
             'Tax Rate', 
             'Pupil-Teacher Ratio',
             'Minority Proportion', 
             'Percent Lower Status', 
             'Median Value']

    longest_stat = max(len(s) for s in stats)
    
    mins = np.amin(data, axis = 0)
    maxs = np.amax(data, axis = 0)
    avgs = np.mean(data, axis = 0)
    stds = np.std(data, axis = 0)
    
    fmt = "{:"+str(longest_stat)+"s} :  {:9s}  |  {:9s}  |  {:9s}  |  {:9s} "
    header = fmt.format("Stat Name", "Minimum", "Maximum", "Average", "Stddev")
    div_str = "-"*len(header)
    val_fmt = "{:9.3f}"
    
    print("\n")
    print(header)
    print(div_str)
    for idx, name in enumerate(stats):
        min_str = val_fmt.format(mins[idx])
        max_str = val_fmt.format(maxs[idx])
        avg_str = val_fmt.format(avgs[idx])
        std_str = val_fmt.format(stds[idx])

        print(fmt.format(name, min_str, max_str, avg_str, std_str))
    print(div_str)
    print("\n")


