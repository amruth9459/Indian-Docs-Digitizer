import streamlit as st
import os
import tempfile
import zipfile
import io
import nest_asyncio
import re
import pytesseract
from pytesseract import Output
from thefuzz import fuzz, process
from llama_parse import LlamaParse
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
import time

# 1. SYSTEM SETUP
nest_asyncio.apply()
st.set_page_config(page_title="Forensic Multilingual Digitizer", page_icon="⚖️", layout="wide")

if "processed_data" not in st.session_state: st.session_state.processed_data = {}
if "review_queue" not in st.session_state: st.session_state.review_queue = []

# 2. LANGUAGE DETECTION HELPER
def contains_foreign_script(text):
    # Detects non-ASCII characters (common in Telugu/Hindi/etc)
    return any(ord(char) > 127 for char in text)

# 3. THE MULTI-ROUND JUDGE
def consult_gemini_forensic(image_crop, ai_guess, scan_guess, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Determine if we need the multilingual prompt
        is_multilingual = contains_foreign_script(scan_guess)
        
        prompt = f"""
        TRANSCRIPTION TASK:
        Image content: Document fragment.
        AI Proposal: "{ai_guess}"
        OCR Scan: "{scan_guess}"
        
        INSTRUCTIONS:
        1. If you see Indian script (Telugu/Hindi), transcribe it exactly.
        2. If the text is physically unreadable (faded/destroyed), output exactly "[ILLEGIBLE]".
        3. Do NOT hallucinate names. If you only see "Pega", write "Pega".
        4. Output ONLY the text.
        """
        
        response = model.generate_content([prompt, image_crop])
        return response.text.strip()
    except:
        return "[ERROR_CALLING_JUDGE]"

# 4. REPLACEMENT LOGIC (Prevents stuttering)
def safe_replace_entity(text, old_val, new_val):
    if not new_val or new_val == old_val: return text
    # Prevents replacing "Pega" with "Pegasus" if "Pegasus" is already there
    if new_val in text and old_val in new_val: return text
    return re.sub(r'\b' + re.escape(old_val) + r'\b', new_val, text)

# 5. MAIN PIPELINE
def run_pipeline(file_path, lk, gk, live_view):
    logs = []
    
    live_view.info("🔄 Round 1: LlamaParse Intelligence...")
    parser = LlamaParse(api_key=lk, result_type="markdown")
    docs = parser.load_data(file_path)
    final_text = "\n\n".join([d.text for d in docs])

    live_view.info("🔄 Round 2: Tesseract Spatial Mapping...")
    pages = convert_from_path(file_path, dpi=300) if file_path.lower().endswith(".pdf") else [Image.open(file_path)]
    
    full_ocr_data = []
    for pg_idx, pg in enumerate(pages):
        data = pytesseract.image_to_data(pg, output_type=Output.DICT)
        for i, word in enumerate(data['text']):
            if len(word.strip()) > 1:
                full_ocr_data.append({
                    "text": word, "page": pg_idx, 
                    "box": (data['left'][i], data['top'][i], data['width'][i], data['height'][i]), 
                    "img": pg
                })
    raw_blob = " ".join([x["text"] for x in full_ocr_data])

    live_view.info("🔄 Round 3: Visual Audit...")
    ai_entities = set(re.findall(r'\b[A-Z][a-z]{2,}\b', final_text))
    
    manual_needed = []

    for word in ai_entities:
        if word.upper() in ["BEFORE", "COURT", "INDIA", "STAMP", "DATE"]: continue
        
        if fuzz.partial_ratio(word.lower(), raw_blob.lower()) < 80:
            # Find closest box to see what happened
            best_match = None
            best_score = 0
            for item in full_ocr_data:
                s = fuzz.ratio(word.lower(), item["text"].lower())
                if s > best_score:
                    best_score = s
                    best_match = item
            
            if best_match:
                x, y, w, h = best_match["box"]
                crop = best_match["img"].crop((max(0, x-10), max(0, y-10), x+w+10, y+h+10))
                
                verdict = consult_gemini_forensic(crop, word, best_match["text"], gk)
                
                # FLAG FOR HUMAN REVIEW IF:
                # 1. Gemini says it's illegible
                # 2. It's a foreign script Gemini isn't 100% sure about
                # 3. AI and Judge completely disagree
                if "[ILLEGIBLE]" in verdict or contains_foreign_script(verdict):
                    manual_needed.append({"word": word, "verdict": verdict, "crop": crop, "page": best_match["page"]})
                else:
                    final_text = safe_replace_entity(final_text, word, verdict)
                    logs.append(f"Fixed: {word} -> {verdict}")

    return final_text, manual_needed, logs

# 6. UI
st.title("⚖️ Forensic Human-in-the-Loop Digitizer")
with st.sidebar:
    lk = st.text_input("LlamaCloud Key", type="password")
    gk = st.text_input("Gemini Key", type="password")
    if st.session_state.processed_data:
        st.divider()
        st.success("Processing Complete")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for n, c in st.session_state.processed_data.items():
                zf.writestr(os.path.splitext(n)[0]+".md", c)
        st.download_button("📥 Download All Results", zip_buffer.getvalue(), "final_docs.zip", "application/zip")

tab1, tab2 = st.tabs(["📤 Upload & Process", "🔍 Manual Review Queue"])

with tab1:
    files = st.file_uploader("Upload Scans", accept_multiple_files=True)
    if st.button("🚀 Start Analysis"):
        if not lk or not gk:
            st.error("Please provide both API keys in the sidebar.")
            st.stop()
            
        live_window = st.empty()
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                path = tmp.name
            
            res, manual, logs = run_pipeline(path, lk, gk, live_window)
            
            if manual:
                st.session_state.review_queue.append({"name": f.name, "text": res, "items": manual})
                st.warning(f"⚠️ {f.name} requires manual review. Check the 'Manual Review Queue' tab.")
            else:
                st.session_state.processed_data[f.name] = res
                st.success(f"✅ {f.name} processed successfully!")
        
        if not st.session_state.review_queue:
            st.success("Batch Complete! All files processed automatically.")
        else:
            st.info("Batch Complete. Some files require manual review.")

with tab2:
    if st.session_state.review_queue:
        for idx, file_review in enumerate(st.session_state.review_queue):
            st.subheader(f"Reviewing: {file_review['name']}")
            for i, item in enumerate(file_review['items']):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(item['crop'], caption="Visual Proof")
                with col2:
                    st.write(f"**AI Suspect:** `{item['word']}`")
                    st.write(f"**Judge Verdict:** `{item['verdict']}`")
                    user_fix = st.text_input(f"Correct this word (or leave as is):", value=item['verdict'], key=f"fix_{idx}_{i}")
                    if st.button(f"Apply Correction", key=f"btn_{idx}_{i}"):
                        file_review['text'] = safe_replace_entity(file_review['text'], item['word'], user_fix)
                        st.success("Updated!")
            
            if st.button(f"✅ Finalize {file_review['name']}", key=f"fin_{idx}"):
                st.session_state.processed_data[file_review['name']] = file_review['text']
                st.session_state.review_queue.pop(idx)
                st.rerun()
    else:
        st.info("No words flagged for manual review.")
