import os 
import pandas as pd
import joblib
from prediction_model.config import config

def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    return _data 

#Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)#joblib.dump(model.state_dict(), 'model_state_dict.joblib')
    print(f"Model has been saved under the name {config.MODEL_NAME}")
#Deserialization
def load_pipeline(pipeline_to_load):
    load_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(load_path)#joblib.dump(model.state_dict(), 'model_state_dict.joblib')
    print("Model has been loaded")
    return model_loaded


