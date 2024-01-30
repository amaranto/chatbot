import logging
import pandas as pd
import matplotlib as plt
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelWithLMHead, GPT2LMHeadModel, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, XLMRobertaTokenizer
from libs.datasets.BaseDataset import BaseDataset

logger = logging.getLogger("NerDataset")

DEFAULT_TOKENIZER_CHECKPOINT="vblagoje/bert-english-uncased-finetuned-pos"
DEFAULT_MODEL_CHECKPOINT="vblagoje/bert-english-uncased-finetuned-pos"
FILTER_BY=["VERB", "NOUN", "PROPN", "ADJ" ]

class NerDataset(BaseDataset):
    def __init__(self, 
                 tokenizer: XLMRobertaTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_CHECKPOINT), 
                 model: AutoModelForTokenClassification |PreTrainedModel  = AutoModelForTokenClassification.from_pretrained(DEFAULT_MODEL_CHECKPOINT)
        ):
        super().__init__(tokenizer, model)
        
        self.filter_by = FILTER_BY
        self.metadata_and_texts : dict = {
            "data": [],
            "summary": {}
        }

    
    def generate_enteties(self, texts: list = [], 
                            min_score: float = 0.7, 
                            grouped_entities: bool = True, 
                            filter_by: list[str] = [],
                            extra_meta_labels: list[str] = [] ):

        if not texts and not self.raw_inputs:
            raise Exception("List of text is required")
        
        if not filter_by:
            filter_by = self.filter_by
        
        entities =  self.pipeline("ner", texts=texts, args={"grouped_entities": grouped_entities})

        summary = {}
        #foo = lambda x : [entity for entity in x if "score" in entity and entity["score"] > min_score]
        def filter_and_summary(x) -> dict:
                result = {}
                for entity in x:
                    if entity and entity["score"] > min_score and entity["entity_group"] in filter_by:
                        word = entity["word"]
                        entity = entity["entity_group"]
                        result[entity] = word

                        if word not in summary:
                            summary[word] = {
                                entity: 1
                            }
                        else:  
                            if entity not in summary[word]:
                                summary[word][entity] = 1
                            else:
                                summary[word][entity] += 1
                return result
        
        self.metadata_and_texts = {
            "data" : [ 
                {
                    "text": entity["text"],
                    "label": entity["label"] if "label" in entity else None,
                    "extra_meta_labels": extra_meta_labels,
                    "metadata": filter_and_summary(entity["result"])
                } for entity in entities 
            ],
            "summary": summary
        }

        return self.metadata_and_texts
    def to_chroma_documents(self):
        
        doc = {
            "texts": [],
            "metadata": [],
            "ids": []
        }

        for data in self.metadata_and_texts["data"]:
            doc["texts"].append(data["text"])
            doc["metadata"].append(data["metadata"])
            doc["ids"].append(data["label"])

        return doc
        
        
    
    def scores_to_csv(self, path: str, header=False, mode='a'):
        df = {
            "label":[],
            "feature": [], 
        }

        for col in self.filter_by:
            df[ col ] = []

        for data in self.metadata_and_texts["data"]:  
            df["label"].append(data["label"])          
            df["feature"].append(data["text"])
            [ df[col].append(data["metadata"][col]) if col in data["metadata"] else df[col].append(None) for col in self.filter_by  ]
        
        df = pd.DataFrame(df)
        df.to_csv(path, index=False, header=header, mode=mode)
        logger.info(f"Saved scores to {path}")
        return df