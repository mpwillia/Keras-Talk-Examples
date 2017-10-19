
from keras.models import load_model
import os

def print_dataset_info(dataset, name = None):
    if name is None:
        print("Dataset Info")
    else:
        print("{} Dataset Info".format(name))
    print("  Input Shape : {}".format(dataset[0].shape))
    print("  Label Shape : {}".format(dataset[1].shape))
    print()


default_compile_kwargs = {'optimizer' : 'adam',
                          'loss' : 'categorical_crossentropy',
                          'metrics' : ['accuracy']}

def test_models(models, training, testing, 
                compile_kwargs = default_compile_kwargs, 
                model_prefix = None):
    
    models_dir = "./models"
    if model_prefix is None:
        model_prefix = ''
    else:
        model_prefix += '-'

    results = []
    # train them all
    for name, model, overwrite in models:
        
        model_name = model_prefix + name + ".h5"

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(model_path) and not overwrite:
            print("Loading Previously Trained {} Model".format(name))
            model = load_model(model_path)
        else:
            print("Training {} Model".format(name))

            model.compile(**compile_kwargs)

            model.fit(*training, 
                      epochs = 5, 
                      batch_size = 128, 
                      verbose = 1, 
                      validation_split = 0.2)
        
        model.save(model_path)

        test_result = model.evaluate(*testing,
                                     batch_size = 128,
                                     verbose = 0)

        results.append((name, test_result)) 
        print() 
    
    for name, result in results:
        print("{} Results".format(name))
        print("  Cost     : {:7.3f}".format(result[0]))
        print("  Accuracy : {:7.2%}".format(result[1]))
        print




