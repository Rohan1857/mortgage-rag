import os
import fitz  # PyMuPDF
import ollama
import io
from PIL import Image

# Robust Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw_pdfs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "extracted_markdown")

def convert_pdf_to_markdown():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Directory {INPUT_DIR} does not exist. Creating it...")
        os.makedirs(INPUT_DIR, exist_ok=True)
        return

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR}.")
        return

    print("Using Ollama (LightOnOCR-2:1b) for extraction...")

    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")
        
        print(f"\nProcessing {filename}...")
        doc = fitz.open(pdf_path)
        final_markdown = f"# Mortgage Record: {filename}\n\n"
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # The model author recommends rendering pages to a target longest dimension of 1540px.
            # Zooming by 2.0x is standard for perfect quality.
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("png")
            
            print(f"  - Extracting Page {page_num + 1}...")
            
            try:
                # Call Ollama with the LightOnOCR-2 Vision Model
                response = ollama.generate(
                    model='Maternion/LightOnOCR-2:latest',
                    prompt='Convert this page to structured Markdown text.',
                    images=[img_bytes]
                )
                page_text = response['response']
                final_markdown += f"## Page {page_num + 1}\n{page_text}\n\n"
                
            except Exception as e:
                print(f"    Error on page {page_num+1}: {e}")
                
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        print(f"Saved: {output_path}")

    print("\nExtraction complete.")

if __name__ == "__main__":
    convert_pdf_to_markdown()