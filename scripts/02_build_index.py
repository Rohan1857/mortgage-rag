import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def build_database():
    print("Configuring Local Embeddings (qwen3-embedding:0.6b via Ollama)...")
    # Ensure Ollama is running before executing this!
    Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")
    Settings.llm = None # We don't need the LLM just to build the index

    # Robust Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "vector_db")
    data_dir = os.path.join(BASE_DIR, "data", "extracted_markdown")

    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("mortgage_slips")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"No markdown files found in {data_dir}. Run 01_run_ocr.py first.")
        return

    print("Loading Markdown documents...")
    documents = SimpleDirectoryReader(data_dir).load_data()

    print("Embedding documents and saving to ChromaDB...")
    # This will use Settings.embed_model automatically in 0.10+
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("Database built successfully!")

if __name__ == "__main__":
    build_database()
