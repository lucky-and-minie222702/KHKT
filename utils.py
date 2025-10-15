import numpy as np
from PIL import Image, ImageOps
import os
from os import path
from torchvision import transforms
from sacrebleu import corpus_bleu
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import joblib
import json
from transformers import TrainerCallback
import random
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import Trainer
from tqdm import tqdm


class DFDistributor:
    def __init__(self, train_df, test_df, n_splits, seed):
        self.train_df = train_df
        self.test_df = test_df
        self.kfold = KFold(n_splits = n_splits, shuffle = True, random_state = seed)
        self.fold = {}
        for i, (train_idx, test_idx) in enumerate(self.kfold.split(train_df), 1):
            self.fold[i] = (train_df.iloc[train_idx], train_df.iloc[test_idx])

def find_subsequence(input_ids, pattern):
    n, m = len(input_ids), len(pattern)
    for i in range(n - m + 1):
        if input_ids[i:i+m] == pattern:
            return i
    return None

def get_dataloader(dataset, batch_size, shuffle = True):
    def collate_fn(batch):
        return {
            key: torch.stack([item[key] for item in batch], dim = 0)
            for key in batch[0]
        }
        
    def get_num_workers():
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        elif "SLURM_CPUS_ON_NODE" in os.environ:
            return int(os.environ["SLURM_CPUS_ON_NODE"])
        else:
            return 0

    return DataLoader(
        dataset = dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        num_workers = get_num_workers(), 
        persistent_workers = True, 
        pin_memory = False, 
        collate_fn = collate_fn
    )

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(p):
    data = dict({})
    with open(p, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

class ImageUtils:
    @staticmethod
    def get_transform(brightness = 0, contrast = 0, rotation_degree = 0):
        return transforms.Compose([
            transforms.ColorJitter( 
                brightness = brightness,
                contrast = contrast,
            ),
            transforms.RandomRotation(rotation_degree),
        ])
    
    @staticmethod
    def get_img_dict():
        file_ids = os.listdir("data/images")
        file_ids = [path.splitext(s)[0] for s in file_ids if s[-4::] == ".jpg"]
        file_paths = [f"data/images/{s}.jpg" for s in file_ids]

        images = dict({})

        for file_path, file_id in zip(file_paths, file_ids):
            images[file_id] = file_path
            
        return images
    
    @staticmethod
    def change_size(img, target_size):
        fill_color = (0, 0, 0)
        target_h, target_w = target_size
        img_h, img_w = img.size
        img = img.resize((target_w, target_w * img_h // img_w), resample = Image.BICUBIC)
        img_h, img_w = img.size

        w, h = img.size
        target_w, target_h = target_size

        if w > target_w or h > target_h:
            return ImageOps.fit(img, target_size, method = Image.BICUBIC, centering = (0.5, 0.5))
        else:
            return ImageOps.pad(img, target_size, method = Image.BICUBIC, color = fill_color, centering = (0.5, 0.5))
        

class TextUtils:
    @staticmethod
    def norm_text(text, final_char = None):
        text = text.strip()
        text = text.replace("\n", "")
        text = text[0].upper() + text[1::]
        if final_char is not None:
            if text[-1] != final_char:
                text += final_char
        return text
    
    @staticmethod
    def get_scores(predictions, references):
        clean_data = [
            (pred.strip().replace("\n", ""), ref.strip().replace("\n", ""))
            for pred, ref in zip(predictions, references)
        ]

        if len(clean_data) == 0:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "meteor": 0.0,
            }

        clean_preds, clean_refs = zip(*clean_data)

        clean_refs_list = [[ref] for ref in clean_refs]

        def compute_bleu_batch(preds, refs):
            scores = []
            for p, r in zip(preds, refs):
                score = sacrebleu.sentence_bleu(p, [r]).score
                scores.append(score)
            mean_score = sum(scores) / len(scores)
            return mean_score / 100

        bleu = compute_bleu_batch(clean_preds, clean_refs)

        # rouge
        r1_total, r2_total, rl_total = 0, 0, 0
        for pred, refs in zip(clean_preds, clean_refs_list):
            ref = refs[0]
            rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer = True)
            scores = rouge.score(ref, pred)
            r1_total += scores["rouge1"].fmeasure
            r2_total += scores["rouge2"].fmeasure
            rl_total += scores["rougeL"].fmeasure
        n = len(clean_preds)
        rouge1 = r1_total / n
        rouge2 = r2_total / n
        rougeL = rl_total / n

        # meteor
        meteor_total = 0
        for pred, refs in zip(clean_preds, clean_refs_list):
            meteor_total += meteor_score(
                [ref.split() for ref in refs],
                pred.split()
            )
        meteor = meteor_total / n

        return {
            "bleu": bleu,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "meteor": meteor
        }


# utils for model
class ModelUtils: 
    class TrainerSaveLossCallback(TrainerCallback):
        def __init__(self, output_dir = None, output_file = "losses.json"):
            self.output_dir = output_dir
            if self.output_dir is None:
                output_dir = "."
            self.output_file = output_file
            self.loss_data = {"train": [], "eval": []}

        def on_log(self, args, state, control, logs = None, **kwargs):
            if logs is not None:
                if "loss" in logs:
                    self.loss_data["train"].append({"step": state.global_step, "loss": logs["loss"]})
                if "eval_loss" in logs:
                    self.loss_data["eval"].append({"step": state.global_step, "eval_loss": logs["eval_loss"]})

        def on_train_end(self, args, state, control, **kwargs):
            p = f"{self.output_dir}/{self.output_file}"
            with open(p, "w") as f:
                json.dump(self.loss_data, f)
            print(f"Losses saved to {p}")
    
    @staticmethod
    def pad_and_trunc(t, max_len, pad_id, side = "right"):
        pad_l = [pad_id] * max(0, max_len - len(t))
        pad_l = torch.tensor(pad_l, dtype = t.dtype)
        if side == "right":
            return torch.cat((t[:max_len:], pad_l), dim = 0)
        else:
            return torch.cat((pad_l, t[:max_len:]), dim = 0)
    
    @staticmethod
    def get_sentences_from_ids(processor, s, to_numpy = False):
        s = s.detach().cpu().numpy().tolist()
        s = processor.tokenizer.batch_decode(s, skip_special_tokens = True)    
        if to_numpy:
            s = np.array(s)
        return s

    @staticmethod
    def get_scores_from_ids(processor, pred, label):
        pred = ModelUtils.get_sentences_from_ids(processor, pred)
        label = ModelUtils.get_sentences_from_ids(processor, label)
        
        return TextUtils.get_scores(pred, label)
    
    @staticmethod
    def print_trainable_params(model):
        trainable, total = 0, 0
        for _, param in model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        print(f"Trainable params: {trainable:,} | Total params: {total:,} "
            f"({100 * trainable / total:.2f}% trainable)")
        
    @staticmethod
    def get_latest_checkpoint(p):
        checkpoints = [d for d in os.listdir(f"results/{p}") if d.startswith("checkpoint")]
        
        if not checkpoints:
            return None

        checkpoints = list(map(lambda x: int(x.split("-")[1]), checkpoints))
        latest = sorted(checkpoints)[-1]
        return latest
        
    class TestLogger():
        def __init__(self, processor):
            self.processor = processor
            self.scores = {
                "bleu": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "meteor": [],
            }
            self.outputs = {
                "questions": [],
                "predictions": [],
                "labels": [],
            }
            
        def log_per_step(self, quest, pred, label, n_returns):
            quest = ModelUtils.get_sentences_from_ids(self.processor, quest, to_numpy = True)
            pred = ModelUtils.get_sentences_from_ids(self.processor, pred, to_numpy = True).reshape(-1, n_returns)
            label = ModelUtils.get_sentences_from_ids(self.processor, label, to_numpy = True)
            
            print(label)
            print()
            print(pred)
            exit()
            
            self.outputs["questions"].append(quest)
            self.outputs["predictions"].append(pred)
            self.outputs["labels"].append(label)
            
            all_scores = {
                "bleu": [],
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "meteor": [],
            }
            
            for i in range(n_returns):
                scores = TextUtils.get_scores(pred[::, i], label)
                for k, v in scores.items():
                    all_scores[k].append(v)
            
            for k, v in all_scores.items():
                self.scores[k].append(v)
                
        def end(self):
            for k, v in self.outputs.items():
                self.outputs[k] = np.concatenate(v, axis = 0)
                
            for k, v in self.scores.items():
                self.scores[k] = np.mean(v, axis = 0)
                
        @property
        def cur_scores(self):
            cur_scores = self.scores.copy()
            for k, v in cur_scores.items():
                cur_scores[k] = np.max(
                    np.mean(v, axis = 0), axis = -1
                )
            return cur_scores
                
        @property
        def results(self):
            return {
                "outputs": self.outputs,
                "scores": self.scores,
            }
            
        class ResultsReader():
            def __init__(self, dir, checkpoint):
                file_path = f"results/{dir}/checkpoint-{checkpoint}-test.results"
                raw_data = joblib.load(file_path)["outputs"]
                
                self.questions = raw_data["questions"]
                self.labels = raw_data["labels"]
                self.predictions = raw_data["predictions"][::, 0]
                
                norm_func = lambda s:  s[:-1:] if s[-1] == "," else s
                norm_map = lambda a: np.array([norm_func(s) for s in a])
                
                self.questions = norm_map(self.questions)
                self.labels = norm_map(self.labels)
                self.predictions = norm_map(self.predictions)
                
                
class TokenWiseAccuracyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_correct_tokens = 0
        self.total_tokens = 0
        self.ignore_index = -100

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
        )

        # if prediction_loss_only:
        #     return (loss, None, None)

        if labels is None:
            return (loss, logits, None)

        predictions = torch.argmax(logits[0], dim = -1)

        mask = labels != self.ignore_index

        masked_predictions = predictions[mask]
        masked_labels = labels[mask]

        correct_tokens = (masked_predictions == masked_labels).sum().item()
        
        total_tokens = mask.sum().item()

        self.total_correct_tokens += correct_tokens
        self.total_tokens += total_tokens

        return (loss, None, None)

    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix: str = "eval"):
        self.total_correct_tokens = 0
        self.total_tokens = 0

        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if self.total_tokens > 0:
            final_accuracy = self.total_correct_tokens / self.total_tokens
        else:
            final_accuracy = 0.0

        metrics[f"{metric_key_prefix}_token_accuracy"] = final_accuracy
        tqdm.write(f"Token-wise accuracy: {final_accuracy:.4f}")

        self.total_correct_tokens = 0
        self.total_tokens = 0
        
        return metrics