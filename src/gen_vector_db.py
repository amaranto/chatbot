import os
import pandas as pd 
import logging
from libs.datasets.Utils import cleaning
from libs.datasets.NER import NerDataset
from libs.Chroma import ChromaManager
from config import ENV, CHROMADB_PATH, CSV_SOURCE, CSV_NER_PATH

logger = logging.getLogger(__name__)

# Set to False to generate the database
# Set to True to skip the generation and use the existing database
SKIP_CSV_TO_CHROMA = True 

db = ChromaManager(use_local_fs=ENV=="colab", path=CHROMADB_PATH)
db.create_collection("cve")

ds = NerDataset()

batch = 1000
total_batches = 200
total_records = batch * total_batches
add_header = not os.path.isfile(CSV_NER_PATH)
mapper = {"feature": lambda x: cleaning(x, max_length=512).lower()} 

if not SKIP_CSV_TO_CHROMA:
    for i in range(0, total_records, batch):
        tokenized_data_batches = ds.load_csv_data(path=CSV_SOURCE, feature="Description",  truncate_before=i, truncate_after=i+batch, label="Name", mappers=mapper )
        ds.pipeline("ner", args={"grouped_entities": True})
        ds.generate_enteties(extra_meta_labels=["cve", "vulnerability"])
        doc = ds.to_chroma_documents()
        tokenized_data_batches.insert_document("cve", doc)
        ds.scores_to_csv(CSV_NER_PATH, header=add_header)
        add_header = False

'''
Example of search
'''
print("Example of search")
r = db.search(collection_name="cve", query="gain root access in linux kernel", k=10)

print(r.keys())
print(r["ids"])
print(r["distances"])
print(r["metadatas"])
print(r["documents"])
