import torch
import logging
import requests
import re
from libs.datasets.Utils import cleaning
from libs.datasets.Translator import TranslatorDataset, LNG_AVAILABLES
from libs.Chroma import ChromaManager
from libs.Wikidata import Wikidata
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedTokenizerFast, pipeline

DEFAULT_BOT_NAME="Tuia"
LNG_DETECTION_MODEL="papluca/xlm-roberta-base-language-detection"
CHAT_MODEL="HuggingFaceH4/zephyr-7b-beta"
HFACE_API="https://api-inference.huggingface.co/models/"
SYSTEM_CONTEXT_MAX_LENGTH = 512
TRANSLATOR_MAX_TOKEN=200
TEXT_GENERATION_MAX_TOKEN=512
MAX_INPUT_LENGTH = 1024 - SYSTEM_CONTEXT_MAX_LENGTH
DEFAULT_STARTER_ID_MARKER = "<_ID_>"
DEFAULT_ENDER_ID_MARKER = "</_ID_/>"

logger = logging.getLogger(__name__)

DEFAULT_MSG = {
    "system" : f"""
    You are {DEFAULT_BOT_NAME} a security expert who always responds using context in the style of technician.
    """,
    "welcome": f"Hello, I am {DEFAULT_BOT_NAME}. I am a security expert.",
    "extra_info": f"You can use {DEFAULT_STARTER_ID_MARKER} and {DEFAULT_ENDER_ID_MARKER} to search for a specific Document. ",
    "not_supported": "I am sorry, but I still have to learn this lenguage. ",
    "not_found": "I am sorry, but I could not find anything related to your question. ",
    "not_info": "I am sorry, but I could not find any information related to your question."
}

class ChatBot():
    def __init__(
            self,
            db: ChromaManager | None = None,
            lng_detection_model_chk: str = LNG_DETECTION_MODEL,
            chat_model_chk: str = CHAT_MODEL,
            enable_web_improvement: bool = True,
            hface_token: str| None = None, 
            system_context_max_length: int = SYSTEM_CONTEXT_MAX_LENGTH,
            translator_max_token: int = TRANSLATOR_MAX_TOKEN,
            text_generation_max_token: int = TEXT_GENERATION_MAX_TOKEN,
            max_input_length: int = MAX_INPUT_LENGTH,
            starter_id_marker: str | None = None,
            ender_id_marker: str | None = None,
            name: str|None = None
    ) -> None:
        self.history: list[dict] = []
        self.supoort_languages: list[str] = ["en", "es"]
        self.translator = TranslatorDataset()
        self.lng_detection_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast  =  AutoTokenizer.from_pretrained(lng_detection_model_chk)
        self.lng_detection_model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(lng_detection_model_chk)
        self.chat_model_chk = chat_model_chk
        self.device = 0 if torch.cuda.is_available() else -1
        self.chat_pipeline = pipeline("text-generation", model=chat_model_chk, torch_dtype=torch.bfloat16, device=self.device) if not hface_token else None 
        self.hface_token : str | None = hface_token
        self.chroma: ChromaManager | None = db
        self.wikidata: Wikidata|None = Wikidata() if enable_web_improvement else None
        self.system_context_max_length = system_context_max_length
        self.translator_max_token = translator_max_token
        self.text_generation_max_token = text_generation_max_token
        self.max_input_length = max_input_length
        self.starter_id_marker = starter_id_marker if starter_id_marker else DEFAULT_STARTER_ID_MARKER
        self.ender_id_marker = ender_id_marker if ender_id_marker else DEFAULT_ENDER_ID_MARKER
        self.name: str = name if name else DEFAULT_BOT_NAME
        self.welcome_msg: str = DEFAULT_MSG["welcome"].replace(DEFAULT_BOT_NAME, self.name) if name else DEFAULT_MSG["welcome"]
        self.extra_info_msg: str = DEFAULT_MSG["extra_info"].replace(DEFAULT_STARTER_ID_MARKER, self.starter_id_marker).replace(DEFAULT_ENDER_ID_MARKER, self.ender_id_marker)
        self.__system_context_msg = DEFAULT_MSG["system"]

    @property
    def system_context_msg(self):
        return self.__system_context_msg

    @system_context_msg.setter
    def system_context_msg(self, msg: str):
        self.__system_context_msg = msg[:self.system_context_max_length]

    def hface_request(self, payload):
        API_URL = f"{HFACE_API}{self.chat_model_chk}"
        headers = {"Authorization": f"Bearer {self.hface_token}"}

        max_tries=3
        response = None
        for i in range(1,max_tries):
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            elif i == max_tries:
                logger.error(f"Error in HuggingFace API: {response.status_code} - {response.text}")
                return {}

    def generate_hface_payload(self, prompt: str, max_new_tokens: int = 150, do_sample: bool = True, return_full_text: bool = False):
        
        payload = {
            'inputs': prompt,
            "parameters": {
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "return_full_text": return_full_text
                }
            }
        return payload

    def generate_message(self, message: list[dict], max_length: int = -1, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95):

        if max_length == -1:
            max_length = self.text_generation_max_token

        if self.hface_token:
            hface_prompt = "" 
            
            for item in message:
                if item["role"] == "system":
                    hface_prompt = f"""
                    <|system|>
                    {item["content"]}</s>
                    """
                elif item["role"] == "user":
                    hface_prompt += f"""
                    <|user|>
                    {item["content"]}</s>
                    """
            hface_prompt += """
            <|assistant|>"""

            hface_prompt = hface_prompt.lstrip()
            logger.info(f"HFace input {hface_prompt}")
            payload = self.generate_hface_payload(hface_prompt)
            response = self.hface_request(payload)
            return response[0]["generated_text"]
        else:
            prompt = self.chat_pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            outputs = self.chat_pipeline(prompt, max_new_tokens=max_length, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p)
            return outputs[0]["generated_text"]
        
    def detect_language(self, text: str, args: dict = {}) -> str:
        pipe = pipeline("text-classification", model=self.lng_detection_model, tokenizer=self.lng_detection_tokenizer, device=self.device, **args)
        lng = pipe(text)[0]

        if lng and lng["score"] > 0.5:
            return lng["label"]
        elif lng and lng["score"] < 0.5:
            return "en"
        else:
            return "en"
    
    def query(self, question: str, collection_name:str, k: int = 10):
        if not self.chroma:
            logger.error("ChromaManager is not initialized")
            return None 
        r = self.chroma.search(collection_name, question, k=k)
        return r
    
    def welcome(self, lng: str = "en")->str:
        
        intro = " ".join([self.welcome_msg, self.extra_info_msg])

        if lng not in self.supoort_languages:
            return intro + " " + DEFAULT_MSG["not_supported"]
        elif lng == "en":
            return intro
        else:
            translation = self.translator.translate(texts=[ intro ], args={"src_lang": LNG_AVAILABLES["en"], "tgt_lang": LNG_AVAILABLES[lng]})
            translation_result = translation[0]["result"]
            welcome_translated = translation_result[0]["translation_text"] if translation_result and "translation_text" in translation_result[0] else "Welcome !"
            return welcome_translated

    def extract_tag(self, text:str):
        try :
            start = self.starter_id_marker
            end = self.ender_id_marker
            regexPattern = start + '(.+?)' + end
            str_found = re.search(regexPattern, text).group(1)

            return str_found, text.replace(start + str_found + end, str_found)
        except AttributeError:
            logger.info("No tag found")
            return None, text


    def ask(self, collection: str, question: str, k: int = 5):

        doc_id, question = self.extract_tag(question)
        question = cleaning(question, max_length=self.max_input_length).lower()

        idiom : str = self.detect_language(question)
        propmpt : list[dict] = []
        response : str = ""

        logger.info(f"Found Id: {doc_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Idiom: {idiom}")

        if idiom in LNG_AVAILABLES and idiom != "en":
            translation = self.translator.translate(texts=[question], args={"src_lang": LNG_AVAILABLES[idiom], "tgt_lang": LNG_AVAILABLES["en"] , "max_length": self.translator_max_token})
        elif idiom not in self.supoort_languages:
            error_msg = f"Language not supported yet."
            logger.error(error_msg)
            return DEFAULT_MSG["not_supported"]
        else:
            translation = None
            
        if translation and "result" in translation[0]:
            translation_result = translation[0]["result"]
            question_translated = translation_result[0]["translation_text"] if translation_result and "translation_text" in translation_result[0] else question
        else:
            question_translated = question
        
        logger.info(f"Question translated: {question_translated} ")
        
        chroma_result = self.chroma.search(collection, question_translated, k=k) if self.chroma else None 
        logger.info(chroma_result)

        if doc_id:
            query_by_id = self.chroma.search_by_id(collection, doc_id)
            # Replace generic search for specific search
            chroma_result = query_by_id if query_by_id and query_by_id["ids"] else chroma_result 

        context = """
        """

        if chroma_result and chroma_result["ids"]:

            labels = [doc_id]  if doc_id else chroma_result["ids"][0]
            documents = chroma_result["documents"][0]

            if self.wikidata:
                wikidata_properties = [self.wikidata.get_property(label) for label in labels] 
                logger.info(f"WikiData Properties : {wikidata_properties}")

                wikidata_descriptions = [ {"item" : prop[0]["itemLabel"]["value"], "description": prop[0]["itemDescription"]["value"]} for prop in wikidata_properties if prop ]
                logger.info(f"WikiData Descriptions : {wikidata_descriptions}")

                mitre_description = [ self.wikidata.get_cve(label) for label in labels ]
                logger.info(f"MITRE Descriptions : {mitre_description}")

            for item in wikidata_descriptions:
                context += f"""{item["item"]}: {item["description"]}
                """.lstrip()

            for label, description in zip(labels, mitre_description):
                context += f"""{label}: {description}
                """.lstrip()

            for label, document in zip(labels, documents):
                context += f"""{label}: {document}
                """.lstrip()
            
            propmpt = [
                {
                    "role": "system",
                    "content": self.system_context_msg + context
                },
                {
                    "role": "user",
                    "content": question_translated
                }
            ]

            response = self.generate_message(propmpt)

        else:
            response = DEFAULT_MSG["not_info"]

        logger.info(f"Response: {response}")

        tmp_responses: list[str] = []
        final_response: str = response

        for i in range(0, len(response), self.translator_max_token):
            
            tmp_response = response[i:i+self.translator_max_token]
            translation = self.translator.translate(texts=[tmp_response], args={"src_lang": LNG_AVAILABLES["en"], "tgt_lang": LNG_AVAILABLES[idiom], "max_length": len(tmp_response)})
            translation_result = translation[0]["result"] if translation and "result" in translation[0] else None
            logger.info(f"Translation: {translation_result}")
            response_translated = translation_result[0]["translation_text"] if translation_result and "translation_text" in translation_result[0] else response
            logger.info(f"Translated Response: {response_translated}")
            tmp_response = response_translated
            tmp_responses.append(tmp_response)

        final_response = " ".join(tmp_responses)
        self.history.append({"question": question, "response": final_response, "collection": collection, "idiom": idiom})
        return final_response