import logging

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, XLMRobertaTokenizer
from libs.datasets.BaseDataset import BaseDataset

logger = logging.getLogger("TranslatorDataset")

DEFAULT_TOKENIZER_CHECKPOINT="facebook/nllb-200-distilled-600M"
DEFAULT_MODEL_CHECKPOINT="facebook/nllb-200-distilled-600M"
LNG_AVAILABLES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn"
}

class TranslatorDataset(BaseDataset):
    def __init__(self, 
                 tokenizer: XLMRobertaTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_CHECKPOINT), 
                 model: AutoModelForSeq2SeqLM | PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_CHECKPOINT)
        ):
        super().__init__(tokenizer, model)
    
    def translate(self, texts: list[str] = [], min_score: float = 0.5, args: dict = { "src_lang":"eng_Latn", "tgt_lang":"spa_Latn", "max_length": 512 }):
        if not "src_lang" in args:
            args["src_lang"] = "eng_Latn"
        if not "tgt_lang" in args:
            args["tgt_lang"] = "spa_Latn"
        if not "max_length" in args:
            args["max_length"] = 512

        if not texts and not self.raw_inputs:
            raise Exception("List of text is required")
        
        if not texts:
            texts = self.raw_inputs

        entities =  self.pipeline("translation", texts=texts, args=args)
        return entities