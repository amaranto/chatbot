import os
import logging

logging.basicConfig(level=logging.INFO)

ENV=os.environ.get("ENV", "colab").lower()
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "data/chroma_data/")
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT", "8000")
CHROMADB_USER = os.environ.get("CHROMADB_USER", "local")
CHROMADB_PASSWORD = os.environ.get("CHROMADB_PASSWORD", "local")
CHROMADB_ALLOW_RESET = os.environ.get("CHROMADB_ALLOW_RESET", "true").lower() == "true"
HFACE_TOKEN = os.environ.get("HFACE_TOKEN", None)
CSV_SOURCE = os.environ.get("CSV_SOURCE", "data/cve.csv")
CSV_NER_PATH = os.environ.get("CSV_NER_PATH", "data/cve_ner.csv")
