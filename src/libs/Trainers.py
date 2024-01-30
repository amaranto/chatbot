import logging
import tempfile
import torch
import re
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

logger = logging.getLogger("ContextTransformerTrainer")

class ContextTransformerTrainer():
    def __init__(self,
                 checkpoint: str | None = None,
                 tokenizer_type: str = "auto",
                 model_type: str = "auto",
                 texts : List[str] = [],
                 labels : List[int|float|str] = [],
                 model_output_dir: str|None = None,
                 save_model_output_dir: str = "./models/cve-model/saves/",
                 gpu: bool = True,
                 gpu_device: str = "cuda:0",
                 special_tokens: dict = {}
                 ):
        
        self._checkpoints = {
            "auto": "microsoft/DialoGPT-medium",
            "xlm-roberta": "intfloat/multilingual-e5-large",
            "gpt2": "microsoft/DialoGPT-medium"
        }

        self._tokenizer_types = {
            "auto": AutoTokenizer,
            "xlm-roberta": XLMRobertaTokenizer,
            "gpt2": GPT2Tokenizer
        }

        self._model_types = {
            "auto": AutoModelWithLMHead,
            "xlm-roberta": AutoModel,
            "gpt2": GPT2LMHeadModel
        }

        self._tokenizer_type = tokenizer_type
        self._checkpoint = checkpoint if checkpoint else self._checkpoints[tokenizer_type]
        self._model_type = model_type
        self._model_output_dir = model_output_dir
        self._save_model_output_dir = save_model_output_dir
        self._texts: List[str] = texts 
        self._labels: List[int|float|str] = labels
        self._gpu = gpu
        #self._train_texts, self._val_texts, self._train_labels, self._val_labels = None, None, None, None
        self._device = gpu_device if torch.cuda.is_available() and self._gpu else "cpu"

        self._tokenizer : XLMRobertaTokenizer| PreTrainedTokenizer | PreTrainedTokenizerFast = self._tokenizer_types[self._tokenizer_type].from_pretrained(self._checkpoint)
        # if self._tokenizer_type == "gpt2" or self._tokenizer_type == "auto":
        #     self._tokenizer.pad_token = self._tokenizer.eos_token 

        self._model : GPT2LMHeadModel | PreTrainedModel | AutoModelWithLMHead  = self._model_types[self._model_type].from_pretrained(self._checkpoint, device_map = self._device)

        if special_tokens:
            logger.info(f"Adding special tokens {special_tokens}")
            self._tokenizer.add_special_tokens(special_tokens)
            self._model.resize_token_embeddings(len(self._tokenizer))

        self._dataset: Dataset | None = None
        self._trainer: Trainer | None = None

        self._training_args : TrainingArguments = TrainingArguments(
            output_dir=self._model_output_dir if self._model_output_dir else tempfile.mkdtemp(),   
            num_train_epochs=3,             
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=-1,
            learning_rate=5e-5,
            overwrite_output_dir=True,
            no_cuda=False if self._gpu else True,
            seed=42,
            warmup_steps=50,
            weight_decay=0.001,
            logging_dir=None,
            save_steps=500,
            fp16=True if self._gpu else False
        )
        
    @property
    def gpu(self):
        return self._gpu
    
    @gpu.setter
    def gpu(self, val: bool):
        # TO-DO Add cuda checker and config
        self._gpu = val
    
    @property
    def save_model_output_dir(self):
        return self._save_model_output_dir
    
    @save_model_output_dir.setter
    def save_model_output_dir(self, val: str):
        self._save_model_output_dir = val
    
    @property
    def tokenizer(self):       
        return self._tokenizer 
    
    @tokenizer.setter
    def tokenizer(self, tokenizer : str ="gpt2", checkpoint : str|None = None):

        if checkpoint:
            self._checkpoint = checkpoint

        if tokenizer in self._tokenizer_types:
            self._tokenizer_type = tokenizer
            self._tokenizer =  self._tokenizer_types[self._tokenizer_type].from_pretrained(self._checkpoint)           
        else:
            raise Exception(f"Tokenizer not supported {tokenizer}")
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model : str ="gpt2", checkpoint : str|None = None):
        if checkpoint:
            self._checkpoint = checkpoint

        if model in self._model_types:
            self._model_type = model
            self._model =  self._model_types[self._model_type].from_pretrained(self._checkpoint)           
        else:
            raise Exception(f"Model not supported {model}")
        
    @property
    def texts(self):
        return self._texts
    
    @texts.setter
    def texts(self, vals: List[str]):
        self._texts = vals
    
    @property
    def labels(self):
        return self._labels
    @labels.setter
    def labels(self, vals: List[int|float|str]):
        self._labels = vals

    @property
    def training_args(self):
        return self._training_args
    
    @training_args.setter
    def training_args(self, vals: TrainingArguments):
        self._training_args = vals

    @property
    def dataset(self):
        return self._dataset
    @dataset.setter
    def dataset(self, vals:Dataset ):
        self._dataset = vals

    def train(self, dataset: Dataset | None = None, args: TrainingArguments | None = None , resume_from_checkpoint = True):

        if not self._dataset and not dataset:
            raise Exception("Dataset not initialized")
        
        training_data = self._dataset if not dataset else dataset
        training_args = self._training_args if not args else args

        logger.info(f"Training model {self._model}")
        logger.info(f"Training data {training_data}")
        
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=training_data
        )

        self.save(save_tokenizer=True, save_model=False)    
        self._trainer.train( resume_from_checkpoint=resume_from_checkpoint )
        self.save(save_tokenizer=False, save_model=True)
        return self._trainer

    def save(self, save_tokenizer: bool = True, save_model: bool = True):
        if not self._trainer and save_model:
            raise Exception("Model not trained")
        elif not self._model_output_dir:
            raise Exception("Model output dir not defined")
        
        if save_tokenizer:
            logger.info(f"Saving tokenizer to {self._save_model_output_dir}")
            self._tokenizer.save_pretrained(self._save_model_output_dir)
        if save_model:
            logger.info(f"Saving model to {self._save_model_output_dir}")
            self._trainer.save_model(self._save_model_output_dir)    

    # Load dataset into GPU memory
    def to_dataset_loader(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0, collate_fn = None, pin_memory: bool = False):

        loaded_dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        return loaded_dataset