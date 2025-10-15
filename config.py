from base import *
from env import *
import pandas as pd

CAUSAL_SETTINGS = ["qwen", "medgemma"]

DEFAULT_CLASS_CONFIG = {
    "model_class": None,
    "processor_class": None,
    "dataset_class": None,
    "format_data_fn": None,
}

def get_config(setting):
    setting = setting.strip().lower()
    conf = DEFAULT_CLASS_CONFIG
    
    if setting == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor 
        conf.update({
            "model_class": Qwen2_5_VLForConditionalGeneration,
            "processor_class": Qwen2_5_VLProcessor,
            "dataset_class": CausalDataset,
            "format_data_fn": CausalDataFormatter(),
        })
    elif setting == "medgemma":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        conf.update({
            "model_class": AutoModelForImageTextToText,
            "processor_class": AutoProcessor,
            "dataset_class": CausalDataset,
            "format_data_fn": CausalDataFormatter(),
        })
        
    return conf

def get_env(conf):
    class_conf = get_config(conf["setting"])

    model_interface = ModelInterface(
        pretrained_name = conf["pretrained_name"],
        model_class = class_conf["model_class"],
        processor_class = class_conf["processor_class"],
    )
    
    env = TrainingEnvironment(
        train_df = pd.read_csv("data/train.csv"),
        test_df = pd.read_csv("data/test.csv"),
        n_splits = conf["n_splits"],
        dataset_class = class_conf["dataset_class"],
        model_interface = model_interface,
        training_args = conf.get("training_args"),
        seed = conf["seed"],
    )
    
    return env, model_interface, class_conf
