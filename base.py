import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
from yaml import Node
from utils import *
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

class ModelInterface:
    def __init__(
        self, 
        pretrained_name, 
        model_class = None, 
        processor_class = None
    ):  
        self.pretrained_name = pretrained_name
        self.model_class = model_class
        self.processor_class = processor_class
        
        if self.model_class is None:
            self.model_class = AutoModel
        if self.processor_class is None:
            self.processor_class = AutoProcessor

        self.model = self.model_class.from_pretrained(
            pretrained_name,
            dtype = torch.bfloat16,
            # device_map = "auto",
            trust_remote_code = True,
            attn_implementation = "flash_attention_2",
        )
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.processor = self.processor_class.from_pretrained(self.pretrained_name)
        
        # load pretrain vision
        print("Load vision encoder")
        vision_lora_config = LoraConfig(
                r = 16,
                lora_alpha = 16,
                target_modules = ["qkv"],
                bias = "none",
        )
        vision = self.model.model.visual
        vision = get_peft_model(vision, vision_lora_config)
        vision.load_state_dict(torch.load("pretrained_vision.torch"))
        vision.merge_and_unload()
        
    def to_lora(self, **kwargs):
        lora_config = LoraConfig(**kwargs)
        
        for p in self.model.model.visual.merger.parameters():
            p.requires_grad = True

        self.model = get_peft_model(self.model, lora_config)
        
    def infer(self, dl, returns =  ["output"], generation_config = {}):
        inputs = []
        outputs = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dl):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                output = self.model.generate(
                    **batch,
                    **generation_config
                )
                
                if "input" in returns:
                    inputs.append(batch["input_ids"])
                
                if "output" in returns:
                    outputs.append(output)
                    
                    
                if "label" in batch:
                    if "label" in returns:
                        labels.append(batch["labels"])
                    
                label = batch["labels"]   
                label[label == -100] == self.processor.tokenizer.pad_token_id
        
        return inputs, outputs, labels
    
    def get_loss(self, dl):
        losses = []
        self.model.eval()        
        with torch.no_grad():
            pbar = tqdm(dl)
            for batch in pbar:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                losses.append(self.model(**batch).loss.item())
                pbar.set_postfix(loss = np.mean(losses))
        
        return np.mean(losses)
            
    def test(
        self, 
        dl,
        output_dir = None, 
        generation_config = None, 
        format_data_fn = None,
    ):
        if generation_config is None:
            generation_config = {
                "do_sample": False,
            }

        logger = ModelUtils.TestLogger(self.processor)
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(dl)
            for batch in pbar:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                input = batch["input_ids"]
                
                label = batch.pop("labels")
                label[label == -100] == self.processor.tokenizer.pad_token_id
                
                output = self.model.generate(
                    **batch,
                    **generation_config,
                )
                
                if format_data_fn is not None:
                    input, output, label = format_data_fn(self.processor, batch, input, output, label)

                logger.log_per_step(
                    quest = input,
                    pred = output,
                    label = label,
                    n_returns = generation_config.get("num_return_sequences", 1),
                )
        
        logger.end()

        if output_dir is not None:
            joblib.dump(logger.results, f"{output_dir}/test.results")

        return logger


# INSTRUCTION = (
#     "You are a medical vision-language assistant; given an endoscopic image and a clinical "
#     "question that may ask about one or more findings, provide a concise, clinically accurate "
#     "response addressing all parts of the question in natural-sounding medical language as if "
#     "spoken by a doctor in a single sentence."
# )

INSTRUCTION = "You are a medical vision assistant about gastroIntestinal image"

ASSISTANT_TEXT = "<|im_start|>assistant\n"

# dataset	     
class BaseDataset(Dataset):
    # mode = "train" or "infer"
    def __init__(
        self, 
        df, 
        processor, 
        mode, 
        img_size, 
        contain_label = True, 
        transform = None
    ):
        super().__init__()
        self.processor = processor
        self.mode = mode
        self.img_size = img_size
        
        self.transform = transform
        if self.transform is None:
            self.transform = {}
        self.transform = ImageUtils.get_transform(**self.transform)

        self.data = df.to_dict(orient = 'records')
        self.img_dict = ImageUtils.get_img_dict()
        self.contain_label = contain_label
        
        self.index = None
        self.quest = None
        self.ans = None
        self.img = None
        
    def process(self):
        self.quest = self.data[self.index]["question"].strip()
        self.quest = TextUtils.norm_text(
            self.quest,
            final_char = self.quest[-1] if self.quest[-1] in [".", "?"] else "?"
        )

        if self.contain_label:
            self.ans = self.data[self.index]["answer"].strip()    
            self.ans = TextUtils.norm_text(
                self.ans,
                final_char = ".",
            )
        
        self.img = Image.open(self.img_dict[self.data[self.index]["img_id"]]).convert("RGB")
        self.img = ImageUtils.change_size(self.img, self.img_size)
        if self.mode == "train":
            self.img = self.transform(self.img)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        self.index = index
        return self.process()
    
class CausalDataset(BaseDataset):
    def __init__(
        self, 
        df, 
        processor, 
        mode,  
        img_size, 
        full_max_length,
        user_max_length,
        assistant_max_length,
        contain_label = True,
        transform = None
    ):
        super().__init__(df, processor, mode, img_size, contain_label, transform)
        self.full_max_length = full_max_length
        self.user_max_length = user_max_length
        self.assistant_max_length = assistant_max_length
    
    def process(self):
        super().process()
        inp_mes = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.img,
                    },
                    {
                        "type": "text",
                        "text": self.quest,
                    }
                ]
            }
        ]
        
        out_mes = []
        if self.contain_label:
            out_mes = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": self.ans,
                        }
                    ]
                }
            ]
        
        merge_mes = inp_mes + out_mes
        
        self.img, _ = process_vision_info(merge_mes)
            
        if self.mode == "infer":
            inp_text = self.processor.apply_chat_template(inp_mes, tokenize = False, add_generation_prompt = True)
            inp = self.processor(
                text = inp_text,
                images = self.img,
                padding = "max_length",
                truncation = True,
                max_length = self.user_max_length,
                return_tensors = "pt"
            )
            inp["labels"] = self.processor.tokenizer(
                text = self.ans,
                padding = "max_length",
                truncation = True,
                max_length = self.assistant_max_length,
                return_tensors = "pt"
            ).input_ids
            inp = {k: v.squeeze(0) for k, v in inp.items()}
            return inp
        
        merge_text = self.processor.apply_chat_template(merge_mes, tokenize = False, add_generation_prompt = False)
        
        merge = self.processor(
            text = merge_text,
            images = self.img,
            padding = "max_length",
            truncation = True,
            max_length = self.full_max_length,
            return_tensors = "pt"
        )
        merge = {k: v.squeeze(0) for k, v in merge.items()}
        
        if not self.contain_label:
            return merge
        
        assistant_pattern = self.processor.tokenizer.encode(ASSISTANT_TEXT)
        
        assistant_idx = find_subsequence(merge["input_ids"].tolist(), assistant_pattern)
        inp_len = assistant_idx + len(assistant_pattern)
        
        label = merge["input_ids"].clone()
        label[:inp_len:] = -100

        merge["labels"] = label
        merge["labels"] = mask_padding_in_labels(merge["labels"], self.processor.tokenizer.pad_token_id)
            
        return merge
    
    
# format data for test
class BaseDataFormatter():
    def __init__(self):
        self.batch = None
        self.input = None
        self.output = None
        self.label = None
        self.processor = None
        
    def fn(self):
        self.label[self.label == -100] = self.processor.tokenizer.pad_token_id
    
    def __call__(self, processor, batch, input, output, label):
        self.processor = processor
        self.batch = batch
        self.input = input
        self.output = output
        self.label = label
        
        self.fn()
        
        return self.input, self.output, self.label
    
class CausalDataFormatter(BaseDataFormatter):
    def fn(self):
        super().fn()
        self.output = self.output[::, self.input.shape[-1]::]
