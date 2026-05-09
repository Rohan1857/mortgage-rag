import os
import re
import json
import html
import base64
import io
from html.parser import HTMLParser
import fitz  # PyMuPDF
import ollama
import requests
from dotenv import load_dotenv

load_dotenv()

# Robust Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted_markdown")
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted_json")
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "ollama").lower()
OCR_MODEL = os.getenv("OCR_MODEL", "maternion/LightOnOCR-2:latest")
OCR_API_BASE = os.getenv("OCR_API_BASE", "").strip()
OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "120"))
OCR_OUTPUT_MODE = os.getenv("OCR_OUTPUT_MODE", "both").lower()

DEFAULT_OPENAI_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "grok": "https://api.x.ai/v1",
}

def _resolve_openai_base_url(provider):
    if OCR_API_BASE:
        return OCR_API_BASE
    shared_base = os.getenv("OPENAI_BASE_URL", "").strip()
    if shared_base:
        return shared_base
    return DEFAULT_OPENAI_BASE_URLS.get(provider, "")

def _get_provider_api_key(provider):
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
    return os.getenv("OCR_API_KEY", "")

def _image_to_data_url(img_bytes):
    encoded = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def _extract_markdown_from_json(text):
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return text

    if not isinstance(payload, dict):
        return text

    pages = payload.get("pages")
    if isinstance(pages, list) and pages:
        page = pages[0] if isinstance(pages[0], dict) else None
        if page:
            for key in ("raw_markdown", "markdown", "text"):
                value = page.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return text

def _should_write_markdown():
    return OCR_OUTPUT_MODE in {"markdown", "both"}

def _should_write_json():
    return OCR_OUTPUT_MODE in {"json", "both"}

class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self._in_table = False
        self._rows = []
        self._current_row = None
        self._current_cell = None
        self._current_row_has_header = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table = True
            self._rows = []
        elif self._in_table and tag == "tr":
            self._current_row = []
            self._current_row_has_header = False
        elif self._in_table and tag in ("td", "th"):
            self._current_cell = []
            if tag == "th":
                self._current_row_has_header = True

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._current_cell is not None:
            text = "".join(self._current_cell)
            self._current_row.append(text)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            self._rows.append((self._current_row_has_header, self._current_row))
            self._current_row = None
        elif tag == "table" and self._in_table:
            self.tables.append(self._rows)
            self._in_table = False

def _collapse_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def _is_markdown_table_line(line):
    line = line.strip()
    return line.startswith("|") and line.count("|") >= 2

def _parse_markdown_row(line):
    cleaned = line.strip().strip("|")
    parts = [p.strip() for p in cleaned.split("|")]
    if not parts:
        return None
    if all(re.fullmatch(r"-+", p) for p in parts):
        return None
    return [p if p else "NaN" for p in parts]

def _split_markdown_cells(line):
    line = line.replace("\\|", "|")
    cleaned = line.strip().strip("|")
    if not cleaned:
        return []
    return [cell.strip() for cell in cleaned.split("|")]

def _is_separator_line(line):
    cells = _split_markdown_cells(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells)

def _format_markdown_row(cells):
    return "| " + " | ".join(cells) + " |"

def _repair_markdown_row(cells, expected_cols):
    cleaned = [cell.strip() for cell in cells if cell.strip()]
    cleaned = [cell for cell in cleaned if not re.fullmatch(r":?-{2,}:?", cell)]
    if not cleaned:
        return []
    if expected_cols <= 0:
        expected_cols = len(cleaned)

    rows = []
    if len(cleaned) <= expected_cols:
        rows.append(_pad_row(cleaned, expected_cols))
        return rows

    for idx in range(0, len(cleaned), expected_cols):
        chunk = cleaned[idx:idx + expected_cols]
        if any(cell.startswith("### ") or cell.startswith("## ") or cell.startswith("# ") for cell in chunk):
            break
        rows.append(_pad_row(chunk, expected_cols))
    return rows

def _split_cell_text(text):
    text = html.unescape(text).replace("\u00a0", " ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    main_parts = []
    extra_rows = []

    for line in lines:
        if _is_markdown_table_line(line):
            row = _parse_markdown_row(line)
            if row:
                extra_rows.append(row)
        else:
            main_parts.append(line)

    main_text = _collapse_whitespace(" ".join(main_parts))
    if not main_text:
        main_text = "NaN"

    return main_text, extra_rows

def _normalize_cell_text(text):
    text = html.unescape(text).replace("\u00a0", " ")
    text = _collapse_whitespace(text)
    if not text:
        return "NaN"
    return text.replace("|", "\\|")

def _pad_row(row, col_count):
    if len(row) < col_count:
        row = row + ["NaN"] * (col_count - len(row))
    elif len(row) > col_count:
        row = row[:col_count]
    return [_normalize_cell_text(cell) for cell in row]

def _table_html_to_markdown(table_html):
    parser = _TableHTMLParser()
    parser.feed(table_html)
    if not parser.tables:
        return table_html

    rows = parser.tables[0]
    if not rows:
        return ""

    header_row = None
    body_rows_raw = []

    for is_header, row in rows:
        if header_row is None and is_header:
            header_row = row
        else:
            body_rows_raw.append(row)

    if header_row is None:
        header_row = []
        body_rows_raw = [row for _, row in rows]

    expanded_body_rows = []
    for row in body_rows_raw:
        expanded_rows = []
        main_row = []
        extra_rows = []

        for cell in row:
            main_cell, nested_rows = _split_cell_text(cell)
            main_row.append(main_cell)
            extra_rows.extend(nested_rows)

        expanded_rows.append(main_row)
        expanded_rows.extend(extra_rows)
        expanded_body_rows.extend(expanded_rows)

    col_count = max(
        len(header_row),
        max((len(row) for row in expanded_body_rows), default=0),
    )

    if not header_row:
        header_row = [f"Column {i + 1}" for i in range(col_count)]

    header_row = _pad_row(header_row, col_count)
    body_rows = [_pad_row(row, col_count) for row in expanded_body_rows]

    separator = ["---"] * col_count

    lines = [
        "| " + " | ".join(header_row) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def _convert_html_tables_to_markdown(text):
    def _replace_table(match):
        return _table_html_to_markdown(match.group(0))

    return re.sub(r"<table[\s\S]*?</table>", _replace_table, text, flags=re.IGNORECASE)

def _strip_markdown_fence(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def _sanitize_ocr_markdown(text):
    text = _extract_markdown_from_json(text)
    text = _strip_markdown_fence(text)
    text = _convert_html_tables_to_markdown(text)
    text = _repair_markdown_tables(text)
    return text.strip()

def _repair_markdown_tables(text):
    lines = text.splitlines()
    repaired = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if (
            _is_markdown_table_line(line)
            and i + 1 < len(lines)
            and _is_separator_line(lines[i + 1])
        ):
            header_cells = _split_markdown_cells(line)
            expected_cols = len(header_cells)
            header_cells = _pad_row(header_cells, expected_cols)
            repaired.append(_format_markdown_row(header_cells))
            repaired.append(_format_markdown_row(["---"] * expected_cols))
            i += 2

            while i < len(lines):
                row_line = lines[i]
                if not _is_markdown_table_line(row_line):
                    break

                heading_match = re.search(r"(#{2,3} .*)", row_line)
                pending_heading = None
                if heading_match:
                    pending_heading = heading_match.group(1).strip()
                    row_line = row_line[:heading_match.start()].rstrip()

                row_cells = _split_markdown_cells(row_line)
                for row in _repair_markdown_row(row_cells, expected_cols):
                    repaired.append(_format_markdown_row(row))

                i += 1

                if pending_heading:
                    repaired.append("")
                    repaired.append(pending_heading)
                    break

            continue

        repaired.append(line)
        i += 1

    return "\n".join(repaired)

def _looks_like_value(text):
    if not text:
        return False
    patterns = [
        r"\$\s*\d",
        r"\d{1,2}/\d{1,2}/\d{2,4}",
        r"\d+\.\d+%",
        r"\d{1,3}(?:,\d{3})+",
    ]
    return any(re.search(pat, text) for pat in patterns)

def _reduce_cells_to_kv_pairs(cells):
    filtered = []
    for cell in cells:
        cleaned = cell.strip()
        if not cleaned or cleaned.lower() == "nan":
            continue
        if re.fullmatch(r":?-{2,}:?", cleaned):
            continue
        filtered.append(cleaned)

    pairs = []
    idx = 0
    while idx + 1 < len(filtered):
        key = filtered[idx]
        value = filtered[idx + 1]
        if re.search(r"#{2,3}\s+", key) or re.search(r"#{2,3}\s+", value):
            break
        pairs.append([key, value])
        idx += 2
    return pairs

def _normalize_table_rows(header_cells, rows_raw):
    expected_cols = len(header_cells)
    if expected_cols <= 0:
        expected_cols = 2

    if expected_cols > 4 or any(_looks_like_value(cell) for cell in header_cells):
        pairs = []
        pairs.extend(_reduce_cells_to_kv_pairs(header_cells))
        for row in rows_raw:
            pairs.extend(_reduce_cells_to_kv_pairs(row))
        if pairs:
            return ["Item", "Value"], pairs

    normalized_rows = []
    for row in rows_raw:
        if len(row) < expected_cols:
            row = row + ["NaN"] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            overflow = " | ".join(row[expected_cols - 1:]).strip()
            row = row[:expected_cols - 1] + [overflow or "NaN"]
        normalized_rows.append([cell if cell else "NaN" for cell in row])

    return header_cells, normalized_rows

def _parse_markdown_table(lines, start_index):
    header_cells = _split_markdown_cells(lines[start_index])
    header_cells = [cell if cell else "NaN" for cell in header_cells]

    row_lines = []
    index = start_index + 2
    while index < len(lines) and _is_markdown_table_line(lines[index]):
        row_lines.append(lines[index])
        index += 1

    rows_raw = [_split_markdown_cells(line) for line in row_lines]
    header_cells, normalized_rows = _normalize_table_rows(header_cells, rows_raw)

    if not header_cells:
        header_cells = ["Column 1", "Column 2"]

    return {
        "headers": header_cells,
        "rows": normalized_rows,
    }, index

def _markdown_to_json(page_markdown, page_number):
    lines = [line.rstrip() for line in page_markdown.splitlines()]
    sections = []

    current_section = {
        "title": "Untitled",
        "text": "",
        "tables": [],
    }
    text_lines = []

    def flush_section():
        text = "\n".join(text_lines).strip()
        current_section["text"] = text
        if current_section["title"] or current_section["text"] or current_section["tables"]:
            sections.append(dict(current_section))

    idx = 0
    while idx < len(lines):
        line = lines[idx]

        if re.match(r"^#{1,3}\s+", line):
            flush_section()
            text_lines = []
            current_section = {
                "title": line.lstrip("# ").strip(),
                "text": "",
                "tables": [],
            }
            idx += 1
            continue

        if (
            _is_markdown_table_line(line)
            and idx + 1 < len(lines)
            and _is_separator_line(lines[idx + 1])
        ):
            table, idx = _parse_markdown_table(lines, idx)
            current_section["tables"].append(table)
            continue

        text_lines.append(line)
        idx += 1

    flush_section()

    return {
        "page_number": page_number,
        "sections": sections,
        "raw_markdown": page_markdown.strip(),
    }

def _ocr_via_ollama(img_bytes, system_prompt, user_prompt):
    response = ollama.generate(
        model=OCR_MODEL,
        system=system_prompt,
        prompt=user_prompt,
        images=[img_bytes],
        options={"temperature": 0},
    )
    return response.get("response", "")

def _ocr_via_openai_compatible(img_bytes, system_prompt, user_prompt, provider):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for this OCR provider.") from exc

    api_key = _get_provider_api_key(provider)
    base_url = _resolve_openai_base_url(provider)
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    data_url = _image_to_data_url(img_bytes)

    response = client.chat.completions.create(
        model=OCR_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return response.choices[0].message.content or ""

def _ocr_via_gemini(img_bytes, system_prompt, user_prompt):
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError("google-generativeai is required for Gemini OCR.") from exc

    api_key = _get_provider_api_key("gemini")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini OCR.")

    genai.configure(api_key=api_key)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for Gemini OCR.") from exc

    image = Image.open(io.BytesIO(img_bytes))
    model = genai.GenerativeModel(
        model_name=OCR_MODEL,
        system_instruction=system_prompt,
    )
    response = model.generate_content([user_prompt, image])
    return getattr(response, "text", "") or ""

def _extract_hf_text(payload):
    if isinstance(payload, list) and payload:
        if isinstance(payload[0], dict):
            for key in ("generated_text", "text", "summary_text"):
                value = payload[0].get(key)
                if value:
                    return value
        if isinstance(payload[0], str):
            return payload[0]
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "summary_text"):
            value = payload.get(key)
            if value:
                return value
    if isinstance(payload, str):
        return payload
    return ""

def _ocr_via_huggingface(img_bytes, system_prompt, user_prompt):
    api_key = _get_provider_api_key("huggingface")
    if not api_key:
        raise RuntimeError("HUGGINGFACE_API_KEY is required for Hugging Face OCR.")

    api_url = os.getenv(
        "HUGGINGFACE_API_URL",
        f"https://api-inference.huggingface.co/models/{OCR_MODEL}",
    )
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(api_url, headers=headers, data=img_bytes, timeout=OCR_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    return _extract_hf_text(payload)

def render_page_to_target_resolution(page, target_long_edge=1540):
    """
    Renders a PDF page dynamically so its longest dimension is exactly
    target_long_edge (1540px), maintaining aspect ratio as recommended by LightOnOCR.
    """
    rect = page.rect
    width = rect.width
    height = rect.height
    
    # Calculate the scale factor needed to make the longest side target_long_edge
    if width >= height:
        scale = target_long_edge / width
    else:
        scale = target_long_edge / height
        
    # Generate the high-quality matrix with the computed scale
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix)
    return pix.tobytes("png")

def convert_pdf_to_markdown():
    if _should_write_markdown():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    if _should_write_json():
        os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Directory {INPUT_DIR} does not exist. Creating it...")
        os.makedirs(INPUT_DIR, exist_ok=True)
        return

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR}.")
        return

    print(f"Configuring OCR provider '{OCR_PROVIDER}' with model '{OCR_MODEL}'...")

    if OCR_PROVIDER == "ollama":
        try:
            ollama.show(OCR_MODEL)
        except ollama.ResponseError:
            print(f"Model {OCR_MODEL} not found. Pulling it now...")
            ollama.pull(OCR_MODEL)

    default_system_prompt = (
        "You are a strict document data extraction engine. Convert the image into clean GitHub Flavored Markdown (GFM). "
        "Follow these rules exactly:\n"
        "1. NO HTML: Use only markdown pipes (|) for tables.\n"
        "2. TABLE BOUNDARIES: Do NOT merge separate sections. If a new header (like ###) begins, end the current table. Never place a markdown header (#) inside a table cell.\n"
        "3. HORIZONTAL DRIFT: Break side-by-side visual elements into vertical sequential blocks instead of creating ultra-wide rows.\n"
        "4. EMPTY CELLS: Write 'NaN' in visually empty table cells to maintain column alignment."
    )
    structured_ocr_prompt = os.getenv("OCR_SYSTEM_PROMPT", default_system_prompt)
    user_prompt = os.getenv("OCR_USER_PROMPT", "Transcribe this page into Markdown only.")

    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")
        json_output_path = os.path.join(JSON_OUTPUT_DIR, f"{base_name}.json")
        
        print(f"\nProcessing {filename}...")
        doc = fitz.open(pdf_path)
        final_markdown = f"# Mortgage Record: {filename}\n\n"
        page_payloads = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Target the 1540px limit recommended by LightOn OCR developers
            img_bytes = render_page_to_target_resolution(page, target_long_edge=1540)
            
            print(f"  - OCR Processing on Page {page_num + 1}...")
            
            try:
                # Direct API call to the local Ollama vision pipeline with explicit prompt instructions
                if OCR_PROVIDER == "ollama":
                    page_text = _ocr_via_ollama(img_bytes, structured_ocr_prompt, user_prompt)
                elif OCR_PROVIDER in {"openai", "openrouter", "groq", "grok"}:
                    page_text = _ocr_via_openai_compatible(
                        img_bytes,
                        structured_ocr_prompt,
                        user_prompt,
                        OCR_PROVIDER,
                    )
                elif OCR_PROVIDER == "gemini":
                    page_text = _ocr_via_gemini(img_bytes, structured_ocr_prompt, user_prompt)
                elif OCR_PROVIDER == "huggingface":
                    page_text = _ocr_via_huggingface(img_bytes, structured_ocr_prompt, user_prompt)
                else:
                    raise RuntimeError(f"Unsupported OCR_PROVIDER: {OCR_PROVIDER}")
                page_text_clean = _sanitize_ocr_markdown(page_text)

                final_markdown += f"## Page {page_num + 1}\n{page_text_clean}\n\n"
                page_payloads.append(_markdown_to_json(page_text_clean, page_num + 1))
                
            except Exception as e:
                print(f"    Error on page {page_num+1}: {e}")
                
        if _should_write_markdown():
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_markdown)
            print(f"Saved OCR markdown output to: {output_path}")

        doc_payload = {
            "source_file": filename,
            "pages": page_payloads,
        }
        if _should_write_json():
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(doc_payload, f, indent=2, ensure_ascii=True)
            print(f"Saved OCR JSON output to: {json_output_path}")

    print("\nAll extractions complete! Highly structured markdown documents generated with NaN mappings.")

if __name__ == "__main__":
    convert_pdf_to_markdown()