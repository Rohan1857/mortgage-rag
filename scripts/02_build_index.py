import os
import re
import json
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

def extract_global_metadata(text: str) -> dict:
    """
    Scans the beginning of the Markdown text to extract global identifiers
    so they can be stamped onto every chunk.
    """
    metadata = {}
    
    # Extract Account Holder Name
    # Handles Markdown formatting (e.g., varying newlines or spaces after the address prompt)
    name_match = re.search(r"See back for mailing address\s*\n+\s*([^\n]+)", text)
    if name_match:
        # Strip out any trailing markdown spaces or pipes
        metadata["account_holder"] = name_match.group(1).replace("|", "").strip()
    else:
        metadata["account_holder"] = "Unknown Account Holder"

    # Extract Account Number (Handles bold markdown tags like **Account Number**)
    acct_match = re.search(r"Account Number\**\s+([0-9]+)", text, re.IGNORECASE)
    if acct_match:
        metadata["account_number"] = acct_match.group(1).strip()
    else:
        metadata["account_number"] = "Unknown Account Number"
        
    return metadata

def _json_doc_to_text(payload: dict) -> str:
    """
    Extracts the perfectly formatted GFM Markdown directly from the JSON envelope.
    """
    lines = []
    source_file = payload.get("source_file")
    if source_file:
        lines.append(f"# Source: {source_file}\n")
        
    for page in payload.get("pages", []):
        # We completely ignore the arrays and grab the raw string the OCR model generated
        raw_md = page.get("raw_markdown", "")
        if raw_md:
            lines.append(raw_md)
            
    return "\n\n".join(lines)

def _load_json_documents(json_dir: str) -> list:
    """
    Reads the JSON files and converts them into LlamaIndex Document objects.
    """
    documents = []
    for filename in os.listdir(json_dir):
        if not filename.lower().endswith(".json"):
            continue
        path = os.path.join(json_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
                text = _json_doc_to_text(payload)
                documents.append(Document(text=text, metadata={"file_name": filename}))
            except json.JSONDecodeError:
                print(f"Error reading JSON from {filename}. Skipping.")
    return documents

def build_database_with_metadata():
    print("Configuring Local Embeddings (qwen3-embedding:0.6b via Ollama)...")
    Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")
    Settings.llm = None  # No chat model needed for indexing

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "vector_db")
    data_dir = os.path.join(BASE_DIR, "data", "extracted_markdown")
    json_dir = os.path.join(BASE_DIR, "data", "extracted_json")

    # Connect to ChromaDB
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection("mortgage_slips")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = []
    if os.path.exists(json_dir) and os.listdir(json_dir):
        print("Loading structured JSON documents...")
        documents = _load_json_documents(json_dir)
    elif os.path.exists(data_dir) and os.listdir(data_dir):
        print("Loading raw Markdown documents...")
        reader = SimpleDirectoryReader(data_dir)
        documents = reader.load_data()
    else:
        print(f"No OCR files found in {json_dir} or {data_dir}. Run your OCR first.")
        return

    # Apply Metadata Enrichment to the raw documents
    for doc in documents:
        global_meta = extract_global_metadata(doc.text)
        doc.metadata.update(global_meta)
        print(f"Stamping Metadata: {global_meta} onto {doc.metadata.get('file_name')}")

    # Parse structural nodes (they automatically inherit the updated document metadata)
    print("Parsing documents into strict Markdown layout nodes...")
    
    # Restored the MarkdownNodeParser! 
    # This prevents your tables from being chopped in half.
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # Embed and persistent index
    print(f"Embedding {len(nodes)} enriched, layout-preserved nodes into ChromaDB...")
    VectorStoreIndex(nodes, storage_context=storage_context)
    print("✅ Enriched database built successfully!")

if __name__ == "__main__":
    build_database_with_metadata()