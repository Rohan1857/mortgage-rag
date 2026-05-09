# Mortgage RAG

Local OCR + RAG pipeline for mortgage statements. The system extracts structured Markdown + JSON from PDFs, builds a layout-safe Chroma vector index, and answers questions with a lightweight Ollama LLM.

## Features
- OCR PDF pages into Markdown and JSON
- Table repair and HTML-to-Markdown cleanup after OCR
- Layout-safe indexing with MarkdownNodeParser
- Local embeddings with ChromaDB
- Streaming chat for fast Q&A

## System Diagram
```mermaid
flowchart TD
    A[Raw PDFs\n data/raw_pdfs] --> B[OCR Script\n scripts/01_run_ocr.py]
    B --> C[Markdown Files\n data/extracted_markdown]
    B --> C2[JSON Files\n data/extracted_json]
    C2 --> D[Index Builder\n scripts/02_build_index.py]
    C --> D
    D --> E[Chroma Vector DB\n vector_db]
    E --> F[Chat App\n app/chat.py]
    F --> G[Answers]

    subgraph Models
        M1[Ollama OCR Model\n Maternion/LightOnOCR-2:latest]
        M2[Ollama Embedding\n qwen3-embedding:0.6b]
        M3[Ollama LLM\n qwen2.5:1.5b]
    end

    M1 --> B
    M2 --> D
    M3 --> F
```

## Setup
1) Create and activate a venv
2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Install and start Ollama, then pull the models:

```bash
ollama pull Maternion/LightOnOCR-2:latest
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5:1.5b
```

## Usage
### 1) OCR PDFs to Markdown + JSON
Place PDFs in `data/raw_pdfs`, then run:

```bash
python scripts/01_run_ocr.py
```

Outputs go to `data/extracted_markdown` and `data/extracted_json`.

### 2) Build the Vector Index
Indexing prefers JSON when present and falls back to Markdown.

```bash
python scripts/02_build_index.py
```

### 3) Start the Chat App

```bash
python app/chat.py
```

## Example
Example question and response after indexing a mortgage statement:

```text
You: What is the account number?
AI: Account number 1234512789090.
```

## Notes
- GPU VRAM is limited; the default models are chosen to stay lightweight.
- OCR emits Markdown and JSON; JSON is preferred for indexing when available.
- Indexing uses a Markdown-aware parser to keep tables intact.
- If OCR quality is poor, try a different Ollama vision model and update `scripts/01_run_ocr.py`.
