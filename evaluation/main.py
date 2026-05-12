import os
import sys
import json
import argparse
import shutil
from dotenv import load_dotenv

# Ensure we can import from the parent mortgage_rag folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import interfaces
from drive_manager import DriveManager
from interfaces import EvaluationResult
from typing import List, Dict, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "eval_checkpoint.json")

def save_checkpoint(stats: Dict, total_semantic_score: float, user_id: str, current_pdf_id: str = None, current_pdf_results: List = None):
    """Saves current stats and state to a local file."""
    checkpoint_data = {
        "user_id": user_id,
        "stats": stats,
        "total_semantic_score": total_semantic_score,
        "current_pdf_id": current_pdf_id,
        "current_pdf_results": [
            (res.model_dump() if hasattr(res, 'model_dump') else res.dict() if hasattr(res, 'dict') else res)
            for res in (current_pdf_results or [])
        ]
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint(current_user_id: str) -> Dict:
    """Loads checkpoint data if it exists and matches the user."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("user_id") == current_user_id:
                    print(f"🔄 Resuming from checkpoint for user: {current_user_id}")
                    return data
                else:
                    print(f"⚠️  Found checkpoint for different user ({data.get('user_id')}). Ignoring.")
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
    return {}

def get_engine(class_name: str):
    """Dynamically instantiates a class from the interfaces module."""
    try:
        cls = getattr(interfaces, class_name)
        return cls()
    except AttributeError:
        raise ImportError(f"Engine class '{class_name}' not found in interfaces.py")

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="🚀 Plug-and-Play RAG Evaluation Harness")
    parser.add_argument("--user", help="Override USER_ID for state tracking")
    parser.add_argument("--ocr", help="Override OCR_ENGINE_CLASS")
    parser.add_argument("--llm", help="Override LLM_ENGINE_CLASS")
    parser.add_argument("--clean", action="store_true", help="Clean temp files before starting")
    return parser.parse_args()

def run_evaluation_pipeline():
    # 0. Configuration & CLI
    load_dotenv()
    args = setup_arg_parser()
    
    user_id = args.user or os.getenv("USER_ID")
    ocr_class = args.ocr or os.getenv("OCR_ENGINE_CLASS", "MortgageOCREngine")
    llm_class = args.llm or os.getenv("LLM_ENGINE_CLASS", "MortgageLLMEngine")

    if not user_id:
        print("❌ Error: USER_ID not found. Set it in .env or use --user")
        return

    if args.clean:
        if os.path.exists("temp_eval"):
            shutil.rmtree("temp_eval")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        print("🧹 Cleaned temp files and checkpoint.")

    # Global Stats Tracking
    checkpoint_data = load_checkpoint(user_id)
    
    if checkpoint_data:
        stats = checkpoint_data["stats"]
        total_semantic_score = checkpoint_data["total_semantic_score"]
    else:
        stats = {
            "total_pdfs": 0,
            "total_questions": 0,
            "total_correct": 0,
            "total_incorrect": 0,
            "accuracy": 0.0,
            "avg_semantic_score": 0.0,
            "accuracy_by_category": {}, # Breakdown per 'type'
            "processed_files": []
        }
        total_semantic_score = 0.0

    os.makedirs("temp_eval", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📊 RAG EVALUATION SIMULATOR")
    print(f"{'='*60}")
    print(f"👤 User:       {user_id}")
    print(f"🔍 OCR Engine: {ocr_class}")
    print(f"🤖 LLM Engine: {llm_class}")
    print(f"{'='*60}\n")

    try:
        drive = DriveManager()
        ocr_engine = get_engine(ocr_class)
        llm_engine = get_engine(llm_class)
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    try:
        pdfs = drive.list_pdfs()
    except Exception as e:
        print(f"❌ Drive Error: {e}")
        return

    if not pdfs:
        print("📭 No PDF files found.")
        return

    try:
        for idx, pdf in enumerate(pdfs, 1):
            pdf_name = pdf['name']
            pdf_id = pdf['id']
            
            # 1. Check local stats first
            local_entry = next((f for f in stats["processed_files"] if f['filename'] == pdf_name), None)
            
            if local_entry:
                if local_entry.get("is_uploaded"):
                    print(f"[{idx}/{len(pdfs)}] ⏭️  Skipping {pdf_name} (Processed & Uploaded)")
                    continue
                else:
                    print(f"[{idx}/{len(pdfs)}] 🔄 {pdf_name} was processed locally but upload failed. Retrying upload...")
                    # We will proceed to the evaluation loop, which will resume from the checkpoint 
                    # if it exists, or re-run if it doesn't. 
                    # If start_idx == len(qa_data), it will skip straight to the upload.
            
            # 2. Check if report already exists on Drive (fallback for quota issues)
            base_name = os.path.splitext(pdf_name)[0]
            if drive.report_exists(f"{base_name}_report.json", user_id):
                print(f"[{idx}/{len(pdfs)}] ⏭️  Skipping {pdf_name} (Report found on Drive)")
                # Sync local stats
                if not local_entry:
                    stats["processed_files"].append({"filename": pdf_name, "questions": 0, "correct": 0, "is_uploaded": True})
                else:
                    local_entry["is_uploaded"] = True
                save_checkpoint(stats, total_semantic_score, user_id)
                continue

            if drive.is_processed_by_user(pdf_id, user_id):
                print(f"[{idx}/{len(pdfs)}] ⏭️  Skipping {pdf_name} (Marked as processed in Drive)")
                if local_entry: local_entry["is_uploaded"] = True
                else: stats["processed_files"].append({"filename": pdf_name, "questions": 0, "correct": 0, "is_uploaded": True})
                continue
                
            print(f"[{idx}/{len(pdfs)}] 🔄 Simulating {pdf_name}...")
            
            try:
                print(f"    📥 Downloading PDF and QA data...")
                pdf_path = drive.download_file(pdf_id)
                qa_path = drive.download_corresponding_qa(pdf_name)
                
                if not qa_path:
                    print(f"    ⚠️  Skipped: QA data (_qa.json) not found for this PDF.")
                    continue

                print(f"    📸 Starting OCR Extraction...")

                with open(qa_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)

                # Resumption logic for the current PDF
                current_pdf_results = []
                start_idx = 0
                if checkpoint_data.get("current_pdf_id") == pdf_id:
                    current_pdf_results_raw = checkpoint_data.get("current_pdf_results", [])
                    current_pdf_results = [EvaluationResult(**res) for res in current_pdf_results_raw]
                    start_idx = len(current_pdf_results)
                    if start_idx > 0:
                        print(f"    ⏩ Resuming {pdf_name} from question {start_idx + 1}")

                # 2. Extract OCR
                print("    📸 Starting OCR Extraction...")
                ocr_results = ocr_engine.extract_text(pdf_path)
                extracted_markdown = ocr_results.get("markdown", "")
                extracted_json = ocr_results.get("json", "")
            
                # 3. Immediate OCR Upload (Ensures data is saved even if evaluation fails)
                print(f"    🚀 Uploading standalone OCR results to Drive...")
                try:
                    # Upload Markdown
                    drive.upload_to_user_folder(
                        content=extracted_markdown,
                        filename=f"OCR_{pdf_name.replace('.pdf', '.md')}",
                        user_id=user_id,
                        mime_type="text/markdown"
                    )
                    
                    # Upload JSON
                    if extracted_json:
                        drive.upload_to_user_folder(
                            content=extracted_json,
                            filename=f"OCR_{pdf_name.replace('.pdf', '.json')}",
                            user_id=user_id,
                            mime_type="application/json"
                        )
                    
                    print("    ✅ Standalone OCR (MD & JSON) uploaded successfully.")
                except Exception as e:
                    print(f"    ⚠️  Note: Standalone OCR upload failed (skipping): {e}")

                # 4. Evaluation Loop
                for res in llm_engine.evaluate_document(ocr_results, qa_data, start_idx=start_idx):
                    current_pdf_results.append(res)
                    
                    # Update Local Stats incrementally
                    stats["total_questions"] += 1
                    total_semantic_score += res.semantic_score
                    if res.is_correct:
                        stats["total_correct"] += 1
                    else:
                        stats["total_incorrect"] += 1
                    
                    # Category Breakdown
                    cat = res.category
                    if cat not in stats["accuracy_by_category"]:
                        stats["accuracy_by_category"][cat] = {"correct": 0, "total": 0, "accuracy": 0.0}
                    stats["accuracy_by_category"][cat]["total"] += 1
                    if res.is_correct:
                        stats["accuracy_by_category"][cat]["correct"] += 1
                    
                    # Save Checkpoint after each question
                    save_checkpoint(stats, total_semantic_score, user_id, current_pdf_id=pdf_id, current_pdf_results=current_pdf_results)

                # Document completed
                correct_in_doc = sum(1 for res in current_pdf_results if res.is_correct)
                total_in_doc = len(current_pdf_results)

                # 3. Finalize and Upload
                stats["total_pdfs"] += 1
                is_uploaded = True
                
                # Prepare individual report
                report_data = [
                    (res.model_dump() if hasattr(res, 'model_dump') else res.dict())
                    for res in current_pdf_results
                ]
                
                try:
                    drive.upload_to_user_folder(json.dumps(report_data, indent=2), f"{base_name}_report.json", user_id, "application/json")
                    drive.mark_as_processed(pdf_id, user_id)
                except Exception as e:
                    if "quota" in str(e).lower():
                        print(f"    ⚠️  Upload failed due to quota. Saving progress locally.")
                        is_uploaded = False
                    else:
                        raise e

                if local_entry:
                    local_entry.update({
                        "questions": total_in_doc,
                        "correct": correct_in_doc,
                        "is_uploaded": is_uploaded
                    })
                else:
                    stats["processed_files"].append({
                        "filename": pdf_name,
                        "questions": total_in_doc,
                        "correct": correct_in_doc,
                        "is_uploaded": is_uploaded
                    })

                # 4. Clear current_pdf from checkpoint as it's fully done
                save_checkpoint(stats, total_semantic_score, user_id, current_pdf_id=None, current_pdf_results=None)
                
                print(f"    ✅ File Complete ({correct_in_doc}/{total_in_doc} correct)")

            except Exception as e:
                err_msg = str(e).lower()
                print(f"    ❌ Error processing {pdf_name}: {e}")
                
                # Detect Rate Limits or Auth Errors to stop the loop
                fatal_keywords = ["rate limit", "429", "resource exhausted", "quota", "invalid_grant"]
                if any(k in err_msg for k in fatal_keywords):
                    print(f"\n🛑 Fatal error detected: {e}. Stopping evaluation to save progress.")
                    break
                
                continue
    except KeyboardInterrupt:
        print("\n\n👋 Evaluation interrupted by user. Generating summary for processed files...")

    # Final Summary & Global Upload
    if stats["total_questions"] > 0:
        stats["accuracy"] = (stats["total_correct"] / stats["total_questions"]) * 100
        stats["avg_semantic_score"] = total_semantic_score / stats["total_questions"]
        
        print(f"\n{'='*60}")
        print(f"🏆 EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Total PDFs:   {stats['total_pdfs']}")
        print(f"❓ Total Qs:     {stats['total_questions']}")
        print(f"✅ Correct:      {stats['total_correct']}")
        print(f"❌ Incorrect:    {stats['total_incorrect']}")
        print(f"🎯 ACCURACY:     {stats['accuracy']:.2f}%")
        print(f"🧠 AVG SEMANTIC: {stats['avg_semantic_score']:.4f}")
        
        if stats["accuracy_by_category"]:
            print(f"\n🏷️  CATEGORY BREAKDOWN:")
            for cat, data in stats["accuracy_by_category"].items():
                data["accuracy"] = (data["correct"] / data["total"]) * 100
                print(f"   - {cat:<12}: {data['accuracy']:>6.2f}% ({data['correct']}/{data['total']})")
        
        print(f"{'='*60}\n")

        # Upload Global Stats to Drive
        print("📤 Uploading overall_accuracy.json...")
        try:
            drive.upload_to_user_folder(json.dumps(stats, indent=2), "overall_accuracy.json", user_id, "application/json")
        except Exception as e:
            if "quota" in str(e).lower():
                print(f"⚠️  Warning: Could not upload overall_accuracy.json due to Drive quota. Stats are saved locally in checkpoint.")
            else:
                print(f"⚠️  Warning: Could not upload overall_accuracy.json: {e}")

        # Clean up checkpoint on successful full completion
        processed_count = len(stats["processed_files"])
        if os.path.exists(CHECKPOINT_FILE) and processed_count >= len(pdfs):
            os.remove(CHECKPOINT_FILE)
            print("🏁 All files processed. Checkpoint removed.")
    else:
        print("\nNo new files were processed.")

if __name__ == "__main__":
    run_evaluation_pipeline()
