import os
import logging
import sys
import requests
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

DEFAULT_OPENAI_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "grok": "https://api.x.ai/v1",
}

# 1. Setup Advanced Logging
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

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

def _build_chat_llm():
    provider = os.getenv("CHAT_PROVIDER", "ollama").lower()
    model = os.getenv("CHAT_MODEL", "qwen2.5:1.5b")
    override_key = os.getenv("CHAT_API_KEY", "")
    override_base = os.getenv("CHAT_API_BASE", "")
    temperature = float(os.getenv("CHAT_TEMPERATURE", "0"))
    request_timeout = float(os.getenv("CHAT_TIMEOUT", "300"))

    if provider == "ollama":
        num_ctx = int(os.getenv("CHAT_NUM_CTX", "4096"))
        return Ollama(
            model=model,
            request_timeout=request_timeout,
            temperature=temperature,
            additional_kwargs={"num_ctx": num_ctx},
        )

    if provider in {"openai", "openrouter", "groq", "grok"}:
        try:
            from llama_index.llms.openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("llama-index-llms-openai is required for this provider.") from exc

        api_key = _get_provider_api_key(provider, override_key)
        api_base = _resolve_openai_base_url(provider, override_base)
        return OpenAI(model=model, api_key=api_key, api_base=api_base or None, temperature=temperature)

    if provider == "gemini":
        try:
            from llama_index.llms.gemini import Gemini
        except ImportError as exc:
            raise RuntimeError("llama-index-llms-gemini is required for Gemini chat.") from exc

        api_key = _get_provider_api_key("gemini", override_key)
        return Gemini(model_name=model, api_key=api_key, temperature=temperature)

    if provider == "huggingface":
        try:
            from llama_index.llms.huggingface import HuggingFaceInferenceAPI
        except ImportError as exc:
            raise RuntimeError("llama-index-llms-huggingface is required for Hugging Face chat.") from exc

        api_key = _get_provider_api_key("huggingface", override_key)
        if not api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY is required for Hugging Face chat.")
        return HuggingFaceInferenceAPI(model_name=model, token=api_key)

    raise RuntimeError(f"Unsupported CHAT_PROVIDER: {provider}")

def start_chat():
    logger.info("Initializing Local RAG Engines...")
    
    # 2. Robust Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "vector_db")

    # 3. Set Models from .env provider configuration
    logger.info("Loading chat and embedding models from .env configuration...")
    Settings.llm = _build_chat_llm()
    Settings.embed_model = _build_embedding_model()

    # 4. Connect to ChromaDB
    if not os.path.exists(db_path):
        logger.error(f"Database folder not found! Run 02_build_index.py first.")
        return

    logger.info("Connecting to ChromaDB Client...")
    db = chromadb.PersistentClient(path=db_path)
    
    try:
        chroma_collection = db.get_collection("mortgage_slips")
    except Exception as e:
        logger.error(f"Collection 'mortgage_slips' not found. Error: {e}")
        return

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 5. Load Index
    logger.info("Building VectorStoreIndex from ChromaDB...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model,
    )

    # 6. Start Chat Engine (Context Mode with Rigid Guardrails)
    logger.info("Warming up the Chat Engine...")
    
    # Advanced System Prompt to explicitly block hallucinations and handle empty cells
    strict_system_prompt = (
        "You are a strict data extraction AI. You analyze structured mortgage documents. "
        "Your task is to answer questions using ONLY the provided text context and metadata. "
        "Observe these rules strictly:\n"
        "1. If you are confused, the data is missing, or the context does not contain the answer, "
        "reply ONLY with: 'I don't know.' Do not attempt to guess.\n"
        "2. Some columns, tables, or fields in the document may be empty or contain blank values. "
        "If asked about these, state clearly that the value is empty or not provided. Never hallucinate any numbers.\n"
        "3. Always check the Metadata block (e.g., account_holder, account_number) prepended to the "
        "retrieved context to ensure you are associating data with the correct customer.\n"
        "4. Be extremely concise. Give the direct answer immediately without conversational filler. "
        "Keep answers under 2 sentences."
    )
    
    chat_engine = index.as_chat_engine(
        chat_mode="context", 
        verbose=False, 
        system_prompt=strict_system_prompt,
    )
    
    print("\n" + "="*60)
    print("✅ RAG System Online (Deterministic Anti-Hallucination Active)")
    print("Type 'exit' to quit")
    print("="*60 + "\n")

    # 7. The Interactive Chat Loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            logger.info("Shutting down RAG system. Goodbye!")
            break
        
        # Simple input validation to prevent empty queries
        if not user_input.strip():
            continue
        
        try:
            logger.info("Retrieving context and generating response...")
            print(f"\nAI: ", end="", flush=True)
            
            # stream_chat prints the words the millisecond the GPU generates them
            response = chat_engine.stream_chat(user_input)
            
            for token in response.response_gen:
                print(token, end="", flush=True)
            print("\n") 
            
        except Exception as e:
            logger.error(f"LLM Generation Failed: {e}", exc_info=True)
            print("\nError: Could not get a response. Check your terminal logs above.")

if __name__ == "__main__":
    start_chat()