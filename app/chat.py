import os
import logging
import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 1. Setup Advanced Logging
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def start_chat():
    logger.info("Initializing Local RAG Engines...")
    
    # 2. Robust Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "vector_db")

    # 3. Set Models (Configured for strict VRAM limits & zero creativity)
    logger.info("Loading Qwen models into LlamaIndex Settings...")
    Settings.llm = Ollama(
        model="qwen2.5:1.5b", 
        request_timeout=300.0,
        temperature=0.0,  # CRITICAL: Forces deterministic, factual extractions. Zero hallucination.
        additional_kwargs={"num_ctx": 4096}  # Safe memory headroom for 1.5B model
    )
    Settings.embed_model = OllamaEmbedding(
        model_name="qwen3-embedding:0.6b",
        request_timeout=300.0
    )

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