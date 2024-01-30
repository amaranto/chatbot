import chromadb 
import logging
from chromadb.utils import embedding_functions
from chromadb.config import Settings

logger = logging.getLogger(__name__)

EMBED_MODEL = "vblagoje/bert-english-uncased-finetuned-pos"

class ChromaManager():
    def __init__(self, 
                 host: str = "localhost", 
                 port: str = "8000",
                 user: str = "local",
                 password: str = "local",
                 settings: Settings | None = None,
                 path: str = "chroma_data/",
                 use_local_fs: bool = False,
                 embedding_foo = None):
        
        self.settings = settings if settings else Settings(
                    chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                    chroma_client_auth_credentials=f"{user}:{password}",
                    allow_reset=True
        )

        self.client = chromadb.PersistentClient(path=path)  if use_local_fs else chromadb.HttpClient(
            host=host,
            port=port,
            settings=self.settings
        )    

        self.collections = {}

        self.embedding_foo = embedding_foo if embedding_foo else embedding_functions.SentenceTransformerEmbeddingFunction(EMBED_MODEL)

        #self.client.heartbeat() # returns a nanosecond heartbeat. Useful for making sure the client remains connected.
        #self.client.reset()

    def create_collection(self, collection_name:str, metadata={"hnsw:space": "cosine"}):
        collection = self.client.get_or_create_collection(
            collection_name,
            metadata=metadata,
            embedding_function=self.embedding_foo
        )

        self.collections[collection_name] = collection

        return collection

    def get_collection(self, collection_name:str):
        if collection_name not in self.collections:
            logger.warning(f"Collection {collection_name} not found. Creating new collection")
            self.create_collection(collection_name)
        return self.collections[collection_name]
    
    def insert_document(self, collection_name:str, documents: dict):
        collection = self.get_collection(collection_name)
        
        collection.add(
            documents=documents["texts"], 
            ids = documents["ids"] if "ids" in documents else [f"id{i}" for i in range(len(documents))],
            metadatas=documents["metadata"] if "metadata" in documents else None,
        )

        return collection

    def search(self, collection_name:str, query: str, k=10):
        collection = self.get_collection(collection_name)
        return collection.query(query_texts=query, n_results=k)
    
    def search_by_id(self, collection_name:str, id: str):
        collection = self.get_collection(collection_name)
        return collection.get(ids=[id])
    
    def reset(self):
        self.client.reset()
