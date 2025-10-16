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
        setting,
        pretrained_name, 
        model_class = None, 
        processor_class = None
    ):  
        self.setting = setting
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
            trust_remote_code = True
        )
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        self.processor = self.processor_class.from_pretrained(self.pretrained_name)
        
    def prepare_for_lora_training(self):
        if self.setting == "qwen":
            for p in self.model.model.visual.merger.parameters():
                p.requires_grad = True
        
    def to_lora(self, **kwargs):
        lora_config = LoraConfig(**kwargs)
        
        self.model = get_peft_model(self.model, lora_config)
        self.prepare_for_lora_training()
        
    def infer(self, dl, returns =  ["output"], generation_config = {}):
        inputs = []
        outputs = []
        labels = []
        
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
                pbar.set_postfix(**{k: round(v, 3) for k, v in logger.cur_scores.items()})
        
        logger.end()

        if output_dir is not None:
            joblib.dump(logger.results, f"{output_dir}/test.results")
            
        for k, v in logger.scores.items():
            print(f"{k}: {np.mean(v):.4f}")

        return logger


# INSTRUCTION = (
#     "You are a medical vision-language assistant; given an endoscopic image and a clinical "
#     "question that may ask about one or more findings, provide a concise, clinically accurate "
#     "response addressing all parts of the question in natural-sounding medical language as if "
#     "spoken by a doctor in a single sentence."
# )

INSTRUCTION = "You are a medical vision assistant about gastroIntestinal image"

ASSISTANT_TEXT = {
    "qwen": "<|im_start|>assistant\n",
}

# dataset	     
class BaseDataset(Dataset):
    # mode = "train" or "infer"
    def __init__(
        self, 
        df, 
        processor, 
        mode, 
        img_size, 
        setting = None,
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
        
        self.setting = setting
        
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
        max_full_length,
        max_user_length,
        max_assistant_length,
        setting = None,
        contain_label = True,
        transform = None
    ):
        super().__init__(df, processor, mode, img_size, setting, contain_label, transform)
        self.max_full_length = max_full_length
        self.max_user_length = max_user_length
        self.max_assistant_length = max_assistant_length
    
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
        
        if self.setting == "qwen":
            self.img, _ = process_vision_info(merge_mes)
            
        if self.mode == "infer":
            inp_text = self.processor.apply_chat_template(inp_mes, tokenize = False, add_generation_prompt = True)
            inp = self.processor(
                text = inp_text,
                images = self.img,
                padding = "max_length",
                truncation = True,
                max_length = self.max_user_length,
                return_tensors = "pt"
            )
            inp["labels"] = self.processor.tokenizer(
                text = self.ans,
                padding = "max_length",
                truncation = True,
                max_length = self.max_assistant_length,
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
            max_length = self.max_full_length,
            return_tensors = "pt"
        )
        merge = {k: v.squeeze(0) for k, v in merge.items()}
        
        if not self.contain_label:
            return merge
        
        assistant_pattern = self.processor.tokenizer.encode(ASSISTANT_TEXT[self.setting])
        
        assistant_idx = find_subsequence(merge["input_ids"].tolist(), assistant_pattern)
        inp_len = assistant_idx + len(assistant_pattern)
        
        label = merge["input_ids"].clone()
        label[:inp_len:] = -100

        merge["labels"] = label
            
        return merge
    
class Seq2seqDataset(BaseDataset):
    def __init__(
        self, 
        df, 
        processor, 
        mode,  
        img_size, 
        q_max_length,
        a_max_length,
        contain_label = True,
        transform = None
    ):
        super().__init__(df, processor, mode, img_size, contain_label, transform)
        self.q_max_length = q_max_length
        self.a_max_length = a_max_length

    def process(self):
        super().process()
        
        self.quest = f"{INSTRUCTION} {self.quest}"
        
        inp = self.processor(
            text = self.quest,
            images = self.img,
            max_length = self.q_max_length,
            padding = "max_length",
            truncation = True,
        )
        
        if self.contain_label:
            label = self.processor.tokenizer(
                text = self.ans,
                max_length = self.a_max_length,
                padding = "max_length",
                truncation = True,
            )["input_ids"]
            label[label == self.processor.tokenizer.pad_token_id] = -100
            inp["labels"] = label
        
        inp = {k: v.squeeze(0) for k, v in inp.items()}
        return inp
    
    
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
    
class Seq2seqDataFormatter(BaseDataFormatter):
    pass