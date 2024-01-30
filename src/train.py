import logging, os
from libs.Trainers import ContextTransformerTrainer
from libs.datasets.BaseDataset import BaseDataset
from libs.datasets.Utils import cleaning

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logging.basicConfig(level=logging.INFO)

model_output_path=f"./models/cve-model-gpt2-custom/"
model_checkpoint_path=f"microsoft/DialoGPT-medium"

trainer = ContextTransformerTrainer(
    model_output_dir=model_output_path, 
    save_model_output_dir=model_output_path,
    special_tokens={"additional_special_tokens": ["<|cve|>"], 'pad_token': '[PAD]'}, 
    checkpoint=model_checkpoint_path, 
    gpu=False
)

tokenizer = trainer.tokenizer
ds = BaseDataset(tokenizer)

if os.path.isfile("data/csv_train_gpt2111.csv"):
    path = "data/csv_train_gpt2.csv"
    save_df_to = None
    mapper = {}
else:
    path = "data/cve.csv"
    save_df_to = "data/csv_train_gpt2111.csv"
    mapper = {"feature": lambda x: "<|cve|>" + cleaning(x, max_length=512) + "<|endoftext|>"}

tokenized_data_batches = ds.load_csv_data(path=path, feature="Description",  truncate_before=0, truncate_after=10, mappers=mapper )
if save_df_to:
    ds.save(save_df_to)
    
ds.encode()
trainer.train(ds, resume_from_checkpoint=False)
trainer.save()