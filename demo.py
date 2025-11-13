from base import *
from env import *
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import sys

model_interface = ModelInterface(
    pretrained_name = sys.argv[1],
    model_class = Qwen2_5_VLForConditionalGeneration,
    processor_class = Qwen2_5_VLProcessor,
)
