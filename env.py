from base import *
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from tqdm import tqdm

class TrainingEnvironment:
    def __init__(
        self, 
        train_df, 
        test_df, 
        n_splits, 
        model_interface: ModelInterface,
        dataset_class = None,
        training_args = None,
        seed = None,
    ):
        if seed is None:
            seed = 27022009
        seed_everything(seed)

        self.distributor = DFDistributor(train_df, test_df, n_splits, seed)
        self.dataset_class = dataset_class
        if self.dataset_class is None:
            self.dataset_class = BaseDataset
        self.model_interface = model_interface
        
        if training_args is None:
            training_args = {}
        self.training_arguments = TrainingArguments(**training_args)
        self.trainer = None
    
    def get_train(self, fold_idx, mode = "train", **kwargs):
        return self.dataset_class(
            df = self.distributor.fold[fold_idx][0],
            processor = self.model_interface.processor,
            mode = mode,
            **kwargs
        )
        
    def get_val(self, fold_idx, mode = "train", **kwargs):
        return self.dataset_class(
            df = self.distributor.fold[fold_idx][1],
            processor = self.model_interface.processor,
            mode = mode,
            **kwargs
        )
        
    def get_test(self, mode = "infer", **kwargs):
        return self.dataset_class(
            df = self.distributor.test_df,
            processor = self.model_interface.processor,
            mode = mode,
            **kwargs
        )
        
    def train(
        self, 
        fold_idx, 
        
        do_train = True,
        do_test = True,
        
        train_ds_args = {},
        val_ds_args = None,
        test_ds_args = None,
        
        lora_args = {},
        
        test_batch_size = 16,
        format_data_fn = None,
        generation_conf = None,
        
        is_causal = True,
        test_output_dir = None,
    ):
        if do_train:
            self.model_interface.processor.tokenizer.padding_side = 'right'

            if val_ds_args is None:
                val_ds_args = train_ds_args

            train_ds = self.get_train(fold_idx, **train_ds_args)
            val_ds = self.get_val(fold_idx, **val_ds_args)
            
            self.model_interface.to_lora(**lora_args)
            
            self.trainer = TokenWiseAccuracyTrainer(
                model = self.model_interface.model,
                args = self.training_arguments,
                processing_class = self.model_interface.processor,
                
                train_dataset = train_ds,
                eval_dataset = val_ds,
                
                callbacks = [
                    ModelUtils.TrainerSaveLossCallback(self.training_arguments.output_dir),
                ],
            )
            
            self.trainer.train()
        
        if do_test:
            if test_ds_args is None:
                test_ds_args = train_ds_args

            if is_causal:
                self.model_interface.processor.tokenizer.padding_side = 'left'
                
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_interface.model.to(device)

            test_ds = self.get_test(mode = "infer", **test_ds_args)
            test_dl = get_dataloader(
                test_ds,
                shuffle = False,
                batch_size = test_batch_size,
            )
            
            res = self.model_interface.test(
                dl = test_dl,
                output_dir = None,
                generation_config = generation_conf,
                format_data_fn = format_data_fn,
            ).results
            
            test_ds = self.get_test(mode = "train", **test_ds_args)
            test_dl = get_dataloader(
                test_ds,
                shuffle = False,
                batch_size = test_batch_size,
            )
            res["loss"] = self.model_interface.get_loss(test_dl)
            
            joblib.dump(res, f"{test_output_dir}/test.results")