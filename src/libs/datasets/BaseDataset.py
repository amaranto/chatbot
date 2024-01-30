import re 
import torch

import pandas as pd
import logging, re
from typing import List, Literal
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import pipeline, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, PreTrainedModel, GPT2LMHeadModel, AutoModelWithLMHead

from libs.datasets.Utils import cleaning
    
logger = logging.getLogger("BaseDataset")

class BaseDataset(Dataset):
    def __init__(self,
                 tokenizer : XLMRobertaTokenizer| PreTrainedTokenizer | PreTrainedTokenizerFast,
                 model : AutoModelForTokenClassification | AutoModelForSeq2SeqLM | PreTrainedModel| GPT2LMHeadModel | AutoModelWithLMHead | None = None,
                 gpu_device: str = "cuda:0"
                 ):
        
        self.raw_inputs : List = [] 
        self.raw_labels : List = [] 
        self.input_ids : List = []
        self.attention_mask : List = []
        self.labels : List =[]
        self.tokenizer : XLMRobertaTokenizer| PreTrainedTokenizer | PreTrainedTokenizerFast = tokenizer
        self.model: AutoModelForTokenClassification | AutoModelForSeq2SeqLM | PreTrainedModel| GPT2LMHeadModel | AutoModelWithLMHead | None = model
        self.device = gpu_device if torch.cuda.is_available() else "cpu"
        self.tokenizer_device = 0 if torch.cuda.is_available() else -1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

    def load_csv_data(self, path: str, feature : str, label : str|None = None, mappers : dict = {}, truncate_before: int = 0, truncate_after: int = -1, on_csv_bad_lines : Literal['error', 'warn', 'skip']="skip"):
        df = pd.read_csv(path, on_bad_lines=on_csv_bad_lines)

        logger.info(f"Dataset loaded {len(df)} rows from {path}")

        if feature not in df.columns:
            raise Exception(f"Feature not found in csv {feature}")
        elif label and label not in df.columns:
            raise Exception(f"Label not found in csv {label}")

        truncate_after = truncate_after if truncate_after >= 0 else len(df)

        df = df.truncate( before=truncate_before, after=truncate_after)

        if "feature" in mappers:
            df[feature] = df[feature].apply(mappers["feature"])
            self.raw_inputs = df[feature].astype(str).tolist()
        else:
            self.raw_inputs = df[feature].astype(str).tolist()
        
        if "label" in mappers:
            df[label] = df[label].apply(mappers["label"])
            self.raw_labels = df[label].tolist()
        elif label:
            self.raw_labels = df[label].tolist()

        result = [ feature, label ] if label else [ feature ]
        return df[result]

        
    def load_tokenized_data(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.labels = tokenized_data["labels"] if "labels" in tokenized_data else tokenized_data["input_ids"]

    def encode(self, texts: list = [], return_tensor: str="pt", use_for_training:bool = True, max_length: int = 512, padding: bool = True, truncation: bool = True, batched:bool = True):

        if not self.tokenizer:
            raise Exception("Tokenizer not initialized")
    
        if not texts and not self.raw_inputs:
            raise Exception("Data not loaded")
        
        raw_data = texts if texts else self.raw_inputs

        result = self.tokenizer(raw_data, truncation=truncation, padding=padding, max_length=max_length, return_tensors=return_tensor)

        if use_for_training:
            self.load_tokenized_data(result)
        
        return result

    def pipeline(self, 
                 name: str, 
                 texts: list = [], 
                 labels: list[str] = [],
                 model: AutoModelForTokenClassification | AutoModelForSeq2SeqLM | PreTrainedModel| GPT2LMHeadModel | AutoModelWithLMHead | None = None,
                 tokenizer: XLMRobertaTokenizer| PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
                 args: dict = {},
                 device: int | None = None
                 ):

        if not model and not self.model:
            raise Exception("Model not initialized")
        elif not tokenizer and not self.tokenizer:
            raise Exception("Tokenizer not initialized")
        elif not texts and not self.raw_inputs:
            raise Exception("Data not loaded")
        
        if not labels:
            labels = self.raw_labels
        
        model = model if model else self.model
        tokenizer = tokenizer if tokenizer else self.tokenizer
        texts = texts if texts else self.raw_inputs

        device = device if device else self.tokenizer_device
        pipe = pipeline(name, model=model, tokenizer=tokenizer, device=device, **args)

        if labels:
            return [{ "text": text, "result": pipe(text), "label": label } for text, label in zip(texts, labels)]
        else:
            return [{ "text": text, "result": pipe(text), "label": None } for text in texts]
        
    def save(self, path: str):

        pd_df = {}

        if self.labels:
            pd_df["labels"] = self.labels
    
        pd_df["feature"] = self.raw_inputs

        df = pd.DataFrame(pd_df)
        df.to_csv(path)

    # def load_json_data(self, path: str):
    #     self.df = pd.read_json(path)
    
    # def load_excel_data(self, path: str):
    #     self.df = pd.read_excel(path)

    # def load_txt_data(self, path: str, sep="\t"):
    #     self.df = pd.read_csv(path, sep=sep)

    # def split_df(self, test_size: float = 0.2, random_state: int = 42):
    #     if not self.texts or not self.labels:
    #         raise Exception("Data not loaded")
    #     elif not self._tokenizer:
    #         raise Exception("Tokenizer not initialized")
        
    #     self._train_texts, self._val_texts, self._train_labels, self._val_labels = train_test_split(self._texts, self._labels, test_size=test_size, random_state=random_state)
    #     return self._train_texts, self._val_texts, self._train_labels, self._val_labels



    
