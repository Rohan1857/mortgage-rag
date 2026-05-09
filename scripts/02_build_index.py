import os
import re
import json
import requests
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

DEFAULT_OPENAI_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "grok": "https://api.x.ai/v1",
}

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

def _resolve_openai_base_url(provider, override):
    if override:
        return override
    shared_base = os.getenv("OPENAI_BASE_URL", "").strip()
    if shared_base:
        return shared_base
    return DEFAULT_OPENAI_BASE_URLS.get(provider, "")

def _get_provider_api_key(provider, override):
    if override:
        return override
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY", "")
    if provider == "groq":
        return os.getenv("GROQ_API_KEY", "")
    if provider == "grok":
        return os.getenv("XAI_API_KEY", "")
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    if provider == "huggingface":
        return os.getenv("HUGGINGFACE_API_KEY", "")
    return ""

def _mean_pool(vectors):
    if not vectors:
        return []
    length = len(vectors[0])
    sums = [0.0] * length
    for vec in vectors:
        for idx, value in enumerate(vec):
            sums[idx] += float(value)
    return [value / len(vectors) for value in sums]

class HuggingFaceInferenceEmbedding(BaseEmbedding):
    def __init__(self, model_name, api_key, api_url=None, timeout=120):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model_name}"
        self.timeout = timeout

    def _request_embedding(self, text):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text}
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and data and isinstance(data[0], list):
            if data and data and isinstance(data[0][0], list):
                return _mean_pool(data[0])
            return data[0]
        if isinstance(data, list) and data and isinstance(data[0], (float, int)):
            return data
        raise ValueError("Unexpected embedding response from Hugging Face API.")

    def _get_query_embedding(self, query):
        return self._request_embedding(query)

    def _get_text_embedding(self, text):
        return self._request_embedding(text)

    def _get_text_embeddings(self, texts):
        return [self._request_embedding(text) for text in texts]

def _build_embedding_model():
    provider = os.getenv("EMBED_PROVIDER", "ollama").lower()
    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    override_key = os.getenv("EMBED_API_KEY", "")
    override_base = os.getenv("EMBED_API_BASE", "")

    if provider == "ollama":
        return OllamaEmbedding(model_name=model, request_timeout=300.0)

    if provider in {"openai", "openrouter", "groq", "grok"}:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as exc:
            raise RuntimeError("llama-index-embeddings-openai is required for this provider.") from exc

        api_key = _get_provider_api_key(provider, override_key)
        api_base = _resolve_openai_base_url(provider, override_base)
        return OpenAIEmbedding(model=model, api_key=api_key, api_base=api_base or None)

    if provider == "gemini":
        try:
            from llama_index.embeddings.gemini import GeminiEmbedding
        except ImportError as exc:
            raise RuntimeError("llama-index-embeddings-gemini is required for Gemini embeddings.") from exc

        api_key = _get_provider_api_key("gemini", override_key)
        return GeminiEmbedding(model_name=model, api_key=api_key)

    if provider == "huggingface":
        api_key = _get_provider_api_key("huggingface", override_key)
        if not api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY is required for Hugging Face embeddings.")
        api_url = os.getenv("HUGGINGFACE_API_URL", "")
        return HuggingFaceInferenceEmbedding(model, api_key, api_url=api_url or None)

    raise RuntimeError(f"Unsupported EMBED_PROVIDER: {provider}")

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
    print("Configuring embedding provider...")
    Settings.embed_model = _build_embedding_model()
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