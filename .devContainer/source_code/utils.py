import pickle
import mlflow
import os

def save(filename:str, obj:object):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename:str) -> object:
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

model_name = "st125171-a3-model"
def load_mlflow(stage='Staging'):
    cache_path = os.path.join("models",stage)
    if(os.path.exists(cache_path) == False):
        os.makedirs(cache_path)
    
    # check if we cache the model
    path = os.path.join(cache_path,model_name)
    if(os.path.exists( path ) == False):
        # This will keep load the model again and again.
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
        save(filename=path, obj=model)

    model = load(path)
    return model