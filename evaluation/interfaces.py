import os
import re
import json
import fitz
import importlib
import math
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# --- DYNAMIC IMPORTS ---
try:
    ocr_m = importlib.import_module("scripts.01_run_ocr")
    render_page_to_image = ocr_m.render_page_to_image
    _sanitize_ocr_markdown = ocr_m._sanitize_ocr_markdown
    _markdown_to_json = ocr_m._markdown_to_json
    _ocr_via_ollama = ocr_m._ocr_via_ollama
    _ocr_via_openai_compatible = ocr_m._ocr_via_openai_compatible
    _ocr_via_gemini = ocr_m._ocr_via_gemini
    _extract_code_blocks = ocr_m._extract_code_blocks

    idx_m = importlib.import_module("scripts.02_build_index")
    extract_global_metadata = idx_m.extract_global_metadata
    _build_embedding_model = idx_m._build_embedding_model
    
    chat_m = importlib.import_module("app.chat")
    _build_chat_llm = chat_m._build_chat_llm

    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import MarkdownNodeParser
except ImportError as e:
    print(f"Warning: Failed to import production scripts: {e}")

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates semantic similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude_v1 or not magnitude_v2: return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

class EvaluationResult(BaseModel):
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    reason: str
    semantic_score: float
    category: Optional[str] = "unknown" # Added to track 'type'

class BaseOCREngine(ABC):
    @abstractmethod
    def extract_text(self, pdf_path: str) -> str: pass

class BaseLLMEngine(ABC):
    @abstractmethod
    def get_answer(self, context: str, question: str) -> str: pass
    @abstractmethod
    def grade_answer(self, q, e, m) -> EvaluationResult: pass

    def evaluate_document(self, context: str, qa_data: Any, start_idx: int = 0):
        """Evaluates a document, yielding results one by one. Supports starting from an offset."""
        # Handle case where qa_data is a dict containing a list of questions
        if isinstance(qa_data, dict):
            for key in ["questions", "data", "items"]:
                if key in qa_data and isinstance(qa_data[key], list):
                    qa_data = qa_data[key]
                    break
            else:
                # If it's a dict but no list key found, treat as single item if valid or error
                if "question" in qa_data:
                    qa_data = [qa_data]
                else:
                    print(f"⚠️  Warning: qa_data is a dict but no 'questions' list found.")
                    return

        if not isinstance(qa_data, list):
            print(f"⚠️  Warning: qa_data is not a list. Type: {type(qa_data)}")
            return

        for i, item in enumerate(qa_data[start_idx:], start=start_idx + 1):
            q = item.get("question")
            e = item.get("answer") or item.get("expected_answer")
            cat = item.get("type") or item.get("category") or "unknown"
            
            if q and e:
                print(f"    [Q{i}/{len(qa_data)}] 📝 Question: {q[:60]}...")
                
                # --- DOCUMENT-LEVEL INDEXING (ONCE PER PDF) ---
                # We build the index only once at the start of the document evaluation
                # to avoid WinError 32 and redundant processing.
                if i == start_idx + 1:
                    print(f"        └─ 🏗️  Building document index...")
                    # Build index logic moved here from get_answer
                    self._current_index = self._build_document_index(context)

                print(f"        └─ 🤖 Answering...")
                ans = self.get_answer(self._current_index, q)
                
                print(f"        └─ ⚖️  Grading...")
                res = self.grade_answer(q, e, ans)
                res.category = cat 
                
                status = "✅ CORRECT" if res.is_correct else "❌ INCORRECT"
                print(f"        └─ {status} (Sim: {res.semantic_score:.2f})")
                yield res

    def _build_document_index(self, context: Any):
        """Builds a LlamaIndex VectorStoreIndex for the given context (MD/JSON)."""
        if isinstance(context, dict):
            markdown_context = context.get("markdown", "")
            json_context = context.get("json", "")
        else:
            markdown_context = str(context)
            json_context = ""

        docs_md = []
        docs_js = []
        if markdown_context:
            doc_md = Document(text=markdown_context)
            # Dynamic metadata extraction if needed
            docs_md.append(doc_md)
        if json_context:
            doc_js = Document(text=json_context)
            docs_js.append(doc_js)

        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
        import chromadb
        
        db_path = os.path.join("temp_eval", "vector_db")
        # Ensure path exists for other temp files, though EphemeralClient won't use it
        os.makedirs(db_path, exist_ok=True)
        
        # Use EphemeralClient (In-Memory) to avoid WinError 32 file-locking on Windows
        db = chromadb.EphemeralClient()
        
        # CRITICAL FIX: The EphemeralClient shares collections across the process.
        # We MUST explicitly delete the old collection to prevent data leakage between PDFs.
        try:
            db.delete_collection("eval_temp")
        except Exception:
            pass # Doesn't exist yet
            
        chroma_collection = db.create_collection("eval_temp")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        chunk_size = int(os.getenv("CHUNK_SIZE", 512))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
        
        nodes = []
        parser_info = []
        if docs_md:
            nodes.extend(MarkdownNodeParser().get_nodes_from_documents(docs_md))
            parser_info.append("md: MarkdownParser")
        if docs_js and json_context.strip():
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes.extend(splitter.get_nodes_from_documents(docs_js))
            parser_info.append(f"json: chunk_size={chunk_size}")
        
        if not nodes:
            raise ValueError("No content nodes generated from OCR.")
            
        print(f"        └─ 📦 Chunked into {len(nodes)} nodes ({', '.join(parser_info)})")
        return VectorStoreIndex(nodes, storage_context=storage_context)

class MortgageOCREngine(BaseOCREngine):
    def __init__(self):
        load_dotenv()
        self.provider = os.getenv("OCR_PROVIDER", "ollama").lower()
        self.sys_p = os.getenv("OCR_SYSTEM_PROMPT", "Convert to Markdown.")
        self.user_p = os.getenv("OCR_USER_PROMPT", "Transcribe.")

    def extract_text(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            full_markdown = ""
            full_json_list = []
            
            for i, page in enumerate(doc):
                img = render_page_to_image(page, dpi=300)
                
                # Internal retry logic for OCR
                try:
                    if self.provider == "ollama": 
                        txt = _ocr_via_ollama(img, self.sys_p, self.user_p)
                        md, js = txt, ""
                    elif self.provider == "gemini": 
                        raw_txt = _ocr_via_gemini(img, self.sys_p, self.user_p)
                        md, js = _extract_code_blocks(raw_txt)
                    else: 
                        txt = _ocr_via_openai_compatible(img, self.sys_p, self.user_p, self.provider)
                        md, js = txt, ""
                except Exception as e:
                    err_msg = str(e).lower()
                    if any(k in err_msg for k in ["rate limit", "429", "resource exhausted", "quota"]):
                        print(f"⚠️  Rate limit hit in OCR ({self.provider}). Waiting 30s...")
                        time.sleep(30)
                        if self.provider == "ollama": 
                            txt = _ocr_via_ollama(img, self.sys_p, self.user_p)
                            md, js = txt, ""
                        elif self.provider == "gemini": 
                            raw_txt = _ocr_via_gemini(img, self.sys_p, self.user_p)
                            md, js = _extract_code_blocks(raw_txt)
                        else: 
                            txt = _ocr_via_openai_compatible(img, self.sys_p, self.user_p, self.provider)
                            md, js = txt, ""
                    else:
                        raise e
                
                full_markdown += _sanitize_ocr_markdown(md) + "\n\n"
                if js:
                    try:
                        full_json_list.append(json.loads(js))
                    except:
                        full_json_list.append({"raw_response": js})
            
            status_msg = f"    ✅ OCR Complete: {len(full_markdown)} characters extracted."
            if any(full_json_list):
                status_msg += " (Structured JSON included)"
            print(status_msg)
            
            return {
                "markdown": full_markdown,
                "json": json.dumps(full_json_list, indent=2) if full_json_list else ""
            }
        except Exception as e:
            raise e

class MortgageLLMEngine(BaseLLMEngine):
    def __init__(self):
        load_dotenv()
        self.key_index = 0
        self.judge_key_index = 0
        self.gemini_key_index = 0
        
        # Cooldown tracking: {api_key: cooldown_expiration_timestamp}
        self.key_cooldowns = {}
        self.COOLDOWN_SECONDS = 2 * 60 # 2 minutes
        self.DAILY_COOLDOWN_SECONDS = 24 * 60 * 60 # 24 hours
        
        # Handle hybrid provider initialization gracefully
        chat_p = os.getenv("CHAT_PROVIDER", "ollama").lower()
        if chat_p == "groq-gemini":
            # Default to None for global settings, logic is handled in get_answer
            Settings.llm = None
        else:
            Settings.llm = _build_chat_llm()
            
        Settings.embed_model = _build_embedding_model()
        
        # Build a separate LLM for grading/judging
        self.judge_llm = self._build_judge_llm()
        
        self.strict_system_prompt = (
            "You are a strict data extraction AI specializing in structured mortgage documents. "
            "### CRITICAL RULE: FOCUS ONLY ON RETRIEVED CONTEXT\n"
            "1. Answer ONLY using the provided text context and metadata. "
            "2. NEVER use your general knowledge, standard industry benchmarks, or standard mortgage context from your training data. "
            "### ADDITIONAL RULES:\n"
            "4. Answer only what is explicitly asked. Do not add extra information.\n"
            "5. Do not show reasoning, chain-of-thought, or analysis. Never output <think> blocks.\n"
            "8. Be extremely concise. Give the direct answer immediately without conversational filler. Keep answers under 2 sentences.\n"
            "9. If the answer is numeric (e.g., dollar amounts, percentages, FICO scores), output ONLY the raw number or the formatted value (e.g., '$1,000,000' or '75%') without any leading/trailing text, explanation, or even a period at the end."
        )

    def _build_judge_llm(self, api_key_override: str = None):
        """Builds a dedicated LLM for grading based on JUDGE_PROVIDER in .env."""
        provider = os.getenv("JUDGE_PROVIDER")
        if not provider or provider == "groq-gemini":
            # Fallback to current CHAT_LLM if no separate judge or hybrid judge is defined
            # In groq-gemini mode, judges are built on-the-fly in grade_answer
            return None
        
        # Temporarily override environment variables to build the judge
        orig_provider = os.getenv("CHAT_PROVIDER")
        orig_model = os.getenv("CHAT_MODEL")
        
        os.environ["CHAT_PROVIDER"] = provider
        os.environ["CHAT_MODEL"] = os.getenv("JUDGE_MODEL", "gpt-oss-120b")
        
        judge = _build_chat_llm(api_key_override=api_key_override)
        
        # Restore original env vars
        if orig_provider: os.environ["CHAT_PROVIDER"] = orig_provider
        if orig_model: os.environ["CHAT_MODEL"] = orig_model
        
        return judge

    def _build_gemini_judge(self):
        """Builds a Gemini LLM as a final fallback for judging."""
        orig_provider = os.getenv("CHAT_PROVIDER")
        orig_model = os.getenv("CHAT_MODEL")
        
        os.environ["CHAT_PROVIDER"] = "gemini"
        os.environ["CHAT_MODEL"] = "gemini-3.1-flash-lite" 
        
        judge = _build_chat_llm()
        
        if orig_provider: os.environ["CHAT_PROVIDER"] = orig_provider
        if orig_model: os.environ["CHAT_MODEL"] = orig_model
        return judge

    def _gemini_genai_backup(self, prompt: str):
        """Standalone backup using rotated Gemini API keys with 24h cooldown tracking."""
        try:
            from google import genai
            from google.genai import types
            
            gemini_keys = [
                os.getenv("GEMINI_API_KEY"),
                os.getenv("GEMINI_API_KEY_1"),
                os.getenv("GEMINI_API_KEY_2"),
                os.getenv("GEMINI_API_KEY_3")
            ]
            valid_keys = [k for k in gemini_keys if k and k.strip()]
            
            if not valid_keys:
                raise RuntimeError("No GEMINI_API_KEY found in .env")

            # Try to find a key not in cooldown
            client = None
            selected_key = None
            for _ in range(len(valid_keys)):
                selected_key = valid_keys[self.gemini_key_index % len(valid_keys)]
                self.gemini_key_index += 1
                
                if time.time() < self.key_cooldowns.get(selected_key, 0):
                    continue
                
                client = genai.Client(api_key=selected_key)
                break
            
            if not client:
                raise RuntimeError("All Gemini keys are currently in cooldown.")

            model = "gemma-4-31b-it"
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH")
            )
            
            # Try the request with retries for 500 errors
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config
                    )
                    return response.text
                except Exception as e:
                    err_msg = str(e).lower()
                    if "500" in err_msg and attempt < 2:
                        print(f"⚠️  Gemini Internal Error (500). Retrying {attempt+1}/3...")
                        time.sleep(2)
                        continue
                        
                    if any(k in err_msg for k in ["rate limit", "429", "resource exhausted", "quota"]):
                        cooldown = self.COOLDOWN_SECONDS
                        if "tokens per day" in err_msg or "daily limit" in err_msg:
                            print(f"🛑 DAILY QUOTA EXCEEDED for Gemini key. Putting in 24h cooldown.")
                            cooldown = self.DAILY_COOLDOWN_SECONDS
                        else:
                            print(f"🚫 Gemini key limited. Putting in 15m cooldown.")
                        
                        self.key_cooldowns[selected_key] = time.time() + cooldown
                        # Retry with next available Gemini key
                        return self._gemini_genai_backup(prompt)
                    raise e
        except Exception as e:
            print(f"❌ Gemini genai SDK backup failed: {e}")
            raise e

    def _rotate_key(self, timeout: float = None):
        """Rotates Groq API keys in a circular manner, skipping those in cooldown."""
        provider = os.getenv("CHAT_PROVIDER", "").lower()
        if provider == "groq":
            keys = [
                os.getenv("GROQ_API_KEY"),
                os.getenv("GROQ_API_KEY_1"),
                os.getenv("GROQ_API_KEY_2"),
                os.getenv("GROQ_API_KEY_3"),
                os.getenv("GROQ_API_KEY_4")
            ]
            valid_keys = [k for k in keys if k and k.strip()]
            if not valid_keys:
                return None
            
            # Try to find a key not in cooldown
            skipped_keys = []
            for _ in range(len(valid_keys)):
                selected_key = valid_keys[self.key_index % len(valid_keys)]
                self.key_index += 1
                
                # Check cooldown
                remaining = int(self.key_cooldowns.get(selected_key, 0) - time.time())
                if remaining > 0:
                    skipped_keys.append(f"{selected_key[:6]}... ({remaining}s)")
                    continue
                    
                # Rebuild the LLM with the new key and optional timeout
                Settings.llm = _build_chat_llm(api_key_override=selected_key, timeout=timeout)
                return selected_key
            
            if skipped_keys:
                print(f"        └─ ⚠️  All Groq keys in cooldown: {', '.join(skipped_keys)}")
        return None

    def get_answer(self, index: Any, question: str) -> str:
        """Retrieves answer using the pre-built index."""
        # Try Groq keys in a circular loop
        groq_keys = [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
            os.getenv("GROQ_API_KEY_4")
        ]
        valid_groq_keys = [k for k in groq_keys if k and k.strip()]
        
        success = False
        ans = ""
        
        if os.getenv("CHAT_PROVIDER") == "groq" and valid_groq_keys:
            model_name = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
            for _ in range(len(valid_groq_keys)):
                current_key = None
                try:
                    current_key = self._rotate_key(timeout=120.0)
                    if not current_key:
                        break # All keys in cooldown
                    
                    top_k = int(os.getenv("TOP_K", 3))
                        
                    # Retrieve context nodes ONCE for all models
                    retriever = index.as_retriever(similarity_top_k=top_k)
                    retrieved_nodes = retriever.retrieve(question)
                    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
                    prompt = f"{self.strict_system_prompt}\n\nRETRIEVED CONTEXT:\n{context_str}\n\nQuestion: {question}"
                    
                    print(f"        └─ 🧬 Top K: {top_k} | Context Nodes: {len(retrieved_nodes)}")

                    # Handle groq/compound specifically using raw client for tools
                    if model_name == "groq/compound":
                        try:
                            from groq import Groq
                        except ImportError:
                            raise ImportError("The 'groq' package is required for 'groq/compound'.")
                            
                        client = Groq(api_key=current_key, timeout=120.0)
                        completion = client.chat.completions.create(
                            model="groq/compound",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_completion_tokens=1024,
                            compound_custom={"tools":{"enabled_tools":["web_search","code_interpreter","visit_website"]}}
                        )
                        ans_raw = completion.choices[0].message.content or ""
                    else:
                        # Use raw Groq client with the pre-built prompt (avoids double retrieval)
                        try:
                            from groq import Groq
                        except ImportError:
                            raise ImportError("The 'groq' package is required.")
                        
                        client = Groq(api_key=current_key, timeout=120.0)
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                            max_completion_tokens=1024
                        )
                        ans_raw = completion.choices[0].message.content or ""
                    
                    # Print token usage
                    usage = getattr(completion, 'usage', None)
                    if usage:
                        print(f"        └─ 🎫 Input Tokens: {usage.prompt_tokens}")
                    else:
                        est_tokens = len(prompt.split()) * 1.3
                        print(f"        └─ 🎫 Input Tokens: ~{int(est_tokens)} (est)")
                    
                    # FORCE FILTER: Remove <think> blocks
                    ans = re.sub(r"<think>.*?</think>", "", ans_raw, flags=re.DOTALL).strip()
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    key_display = f"{current_key[:6]}...{current_key[-4:]}" if current_key else "UNKNOWN"
                    print(f"        └─ ❌ KEY {key_display} FAILED: {err_msg[:100]}...")
                    
                    if current_key:
                        cooldown = self.COOLDOWN_SECONDS
                        if any(k in err_msg for k in ["tokens per day", "daily limit", "daily quota"]):
                             print(f"           🛑 Daily quota hit. 24h cooldown.")
                             cooldown = self.DAILY_COOLDOWN_SECONDS
                        elif "timeout" in err_msg or "deadline" in err_msg:
                             print(f"           ⏲️  Timeout (2m) hit. 2m cooldown.")
                        else:
                             print(f"           🔄 Rotating to next key (2m cooldown)")
                        
                        self.key_cooldowns[current_key] = time.time() + cooldown
                    continue
        
        # Fallback to Gemini if Groq failed or not used
        if not success:
            print("🚫 All Groq keys failed or in cooldown. Falling back to Gemini...")
            try:
                # Build prompt from the index if it wasn't built in the Groq loop
                if 'prompt' not in dir():
                    top_k = int(os.getenv("TOP_K", 3))
                    retriever = index.as_retriever(similarity_top_k=top_k)
                    retrieved_nodes = retriever.retrieve(question)
                    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
                    prompt = f"{self.strict_system_prompt}\n\nRETRIEVED CONTEXT:\n{context_str}\n\nQuestion: {question}"
                
                ans_raw = self._gemini_genai_backup(prompt)
                ans = re.sub(r"<think>.*?</think>", "", ans_raw, flags=re.DOTALL).strip()
                
                # Print token usage estimate for Gemini
                est_tokens = len(prompt.split()) * 1.3
                print(f"        └─ 🎫 Input Tokens: ~{int(est_tokens)} (est)")
                success = True
            except Exception as e:
                print(f"❌ Gemini answering fallback failed: {e}")
                raise e
                
        return ans

    def grade_answer(self, question, expected, model_ans) -> EvaluationResult:
        try:
            # 0. Optimization: Auto-fail if model admits ignorance
            if model_ans.strip().lower() == "i don't know.":
                return EvaluationResult(
                    question=question, expected_answer=expected, model_answer=model_ans,
                    is_correct=False, semantic_score=0.0,
                    reason="Model admitted ignorance (I don't know.)."
                )

            # 1. Calculate Semantic Similarity via Embeddings
            vec_exp = Settings.embed_model.get_text_embedding(expected)
            vec_mod = Settings.embed_model.get_text_embedding(model_ans)
            sim_score = cosine_similarity(vec_exp, vec_mod)

            # Safety Truncation to prevent 413 Request Entity Too Large errors
            q_trunc = question[:2000]
            e_trunc = expected[:2000]
            m_trunc = model_ans[:2000]

            # 2. HYBRID LOGIC
            # AUTO-PASS: High confidence similarity
            if sim_score > 0.92:
                return EvaluationResult(
                    question=question, expected_answer=expected, model_answer=model_ans,
                    is_correct=True, semantic_score=sim_score,
                    reason="Auto-passed: Extremely high semantic similarity."
                )
            
            # ALL OTHER CASES: Send to LLM Judge for factual verification
            prompt = (
                "### ROLE: Robotic Factual Auditor\n"
                "You are a binary validator. Your ONLY job is to verify if the 'Model Answer' matches the 'Expected Answer' (GROUND TRUTH).\n\n"
                "### STRICT RULES:\n"
                "1. **IGNORE REAL-WORLD KNOWLEDGE**: Even if you think the GROUND TRUTH is factually incorrect according to real-world mortgage rules, you MUST treat it as the ONLY truth. Your knowledge of '70%' or 'universal standards' is irrelevant.\n"
                "2. **MATCHING DATA**: If the Model Answer contains the specific value/data point from the Ground Truth, it is CORRECT.\n"
                "3. **NO CRITIQUING**: Do not provide 'corrections' based on external info in your reason. If it matches Ground Truth, it is correct. Period.\n\n"
                f"Question: {q_trunc}\n"
                f"GROUND TRUTH: {e_trunc}\n"
                f"MODEL ANSWER: {m_trunc}\n"
                f"Similarity Hint: {sim_score:.2f}\n\n"
                "Reply ONLY in JSON: {\"is_correct\": bool, \"reason\": \"concise factual reason\"}"
            )
            
            # --- Dedicated Judge with Multi-Key Rotation and Gemini Fallback ---
            cerebras_keys = [
                os.getenv("CEREBRAS_API_KEY"),
                os.getenv("CEREBRAS_API_KEY_1"),
                os.getenv("CEREBRAS_API_KEY_2"),
                os.getenv("CEREBRAS_API_KEY_3")
            ]
            valid_judge_keys = [k for k in cerebras_keys if k and k.strip()]
            
            success = False
            res = None
            
            # 1. Try Primary Judge keys in a circular loop
            judge_provider = os.getenv("JUDGE_PROVIDER", "cerebras").lower()
            
            if judge_provider == "cerebras":
                judge_keys = [os.getenv("CEREBRAS_API_KEY"), os.getenv("CEREBRAS_API_KEY_1"), os.getenv("CEREBRAS_API_KEY_2"), os.getenv("CEREBRAS_API_KEY_3")]
            elif judge_provider == "groq":
                judge_keys = [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3"), os.getenv("GROQ_API_KEY_4")]
            elif judge_provider == "groq-gemini":
                # Combine all 8 keys into a mixed pool
                groq_keys = [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3"), os.getenv("GROQ_API_KEY_4")]
                gemini_keys = [os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2"), os.getenv("GEMINI_API_KEY_3")]
                judge_keys = groq_keys + gemini_keys
            else:
                judge_keys = []
                
            valid_judge_keys = [k for k in judge_keys if k and k.strip()]
            
            if valid_judge_keys:
                for _ in range(len(valid_judge_keys)):
                    key = valid_judge_keys[self.judge_key_index % len(valid_judge_keys)]
                    self.judge_key_index += 1
                    
                    # Check cooldown
                    if time.time() < self.key_cooldowns.get(key, 0):
                        continue

                    try:
                        # Determine provider for this specific key
                        if key.startswith("gsk_"): # Groq Key
                            print(f"    ⚖️  Using Groq Judge Key...")
                            judge_model = _build_chat_llm(api_key_override=key, timeout=120.0)
                            res = judge_model.complete(prompt)
                        elif key.startswith("csk-"): # Cerebras Key
                            print(f"    ⚖️  Using Cerebras Judge Key...")
                            # Force Cerebras provider and model for this call
                            os.environ["CHAT_PROVIDER"] = "cerebras"
                            os.environ["CHAT_MODEL"] = os.getenv("JUDGE_MODEL", "llama3.1-8b")
                            judge_model = _build_chat_llm(api_key_override=key, timeout=120.0)
                            res = judge_model.complete(prompt)
                        else: # Assume Gemini Key
                            print(f"    ⚖️  Using Gemini Judge Key...")
                            orig_key = os.environ.get("GEMINI_API_KEY")
                            os.environ["GEMINI_API_KEY"] = key
                            res = self._gemini_genai_backup(prompt)
                            if orig_key: os.environ["GEMINI_API_KEY"] = orig_key
                        
                        success = True
                        break
                    except Exception as e:
                        err_msg = str(e).lower()
                        key_display = f"{key[:6]}...{key[-4:]}"
                        print(f"    ⚖️  ❌ JUDGE KEY {key_display} FAILED: {err_msg[:100]}...")
                        
                        cooldown = self.COOLDOWN_SECONDS
                        if any(k in err_msg for k in ["tokens per day", "daily limit", "daily quota"]):
                            print(f"       🛑 Daily quota hit. 24h cooldown.")
                            cooldown = self.DAILY_COOLDOWN_SECONDS
                        elif "timeout" in err_msg or "deadline" in err_msg:
                            print(f"       ⏲️  Timeout (2m) hit. 2m cooldown.")
                        else:
                            print(f"       🔄 Rotating to next judge key (2m cooldown).")
                        
                        self.key_cooldowns[key] = time.time() + cooldown
                        continue
            
            # 2. If no Cerebras or all failed, try the current judge_llm (might be fallback already)
            if not success:
                try:
                    res = self.judge_llm.complete(prompt)
                    success = True
                except Exception as e:
                    err_msg = str(e).lower()
                    if any(k in err_msg for k in ["rate limit", "429", "413", "resource exhausted", "quota", "request_too_large"]):
                        print(f"🚫 All primary judge keys failed. Falling back to Gemini (gemma-4-31b-it)...")
                        try:
                            # Use the requested SDK fallback for judging too
                            res_text = self._gemini_genai_backup(prompt)
                            # Convert text back to the expected JSON format if possible, or mock it
                            # The prompt asks for JSON, so we expect Gemini to return it
                            success = True
                            # We'll parse res_text in the next block
                            res = res_text 
                        except Exception as e_gemini:
                            print(f"❌ Gemini fallback also failed: {e_gemini}")
                            raise e_gemini
                    else:
                        raise e

            try:
                match = re.search(r"\{.*\}", str(res), re.DOTALL)
                data = json.loads(match.group()) if match else json.loads(str(res))
                return EvaluationResult(
                    question=question, expected_answer=expected, model_answer=model_ans, 
                    semantic_score=sim_score, **data
                )
            except:
                return EvaluationResult(
                    question=question, expected_answer=expected, model_answer=model_ans, 
                    is_correct=(sim_score > 0.80), semantic_score=sim_score, 
                    reason="LLM parsing failed. Used 0.80 similarity threshold fallback."
                )
        except Exception as e:
            # Re-raise rate limit errors to be caught by main loop for checkpointing
            raise e

# Mocks
class MockOCREngine(BaseOCREngine):
    def extract_text(self, p): return "mock"
class MockLLMEngine(BaseLLMEngine):
    def get_answer(self, c, q): return "mock"
    def grade_answer(self, q, e, m): return EvaluationResult(question=q, expected_answer=e, model_answer=m, is_correct=True, reason="mock", semantic_score=1.0)
