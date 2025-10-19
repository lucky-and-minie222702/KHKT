from base import *
from env import *
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

def get_env(conf):
    model_interface = ModelInterface(
        pretrained_name = conf["pretrained_name"],
        model_class = Qwen2_5_VLForConditionalGeneration,
        processor_class = Qwen2_5_VLProcessor,
    )
    
    env = TrainingEnvironment(
        train_df = pd.read_csv("data/train.csv"),
        test_df = pd.read_csv("data/test.csv"),
        n_splits = conf["n_splits"],
        dataset_class = CausalDataset,
        model_interface = model_interface,
        training_args = conf.get("training_args"),
        seed = conf["seed"],
    )
    
    return env, model_interface
