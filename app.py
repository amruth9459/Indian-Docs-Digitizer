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
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import google.generativeai as genai
import time
from collections import Counter

# 1. SYSTEM SETUP
nest_asyncio.apply()
st.set_page_config(page_title="Industrial Forensic Scaler", page_icon="⚖️", layout="wide")

if "processed_data" not in st.session_state: st.session_state.processed_data = {}
if "review_queue" not in st.session_state: st.session_state.review_queue = []

# 2. UTILS
def contains_foreign_script(text):
    return any(ord(char) > 127 for char in text)

def safe_replace_entity(text, old_val, new_val):
    if not new_val or new_val == old_val: return text
    if new_val in text and old_val in new_val: return text
    return re.sub(r'\b' + re.escape(old_val) + r'\b', new_val, text)

# 3. ENTITY EXTRACTION (Golden Truths)
def extract_global_entities(full_text):
    # Pattern focused on legal entity types common in Indian courts
    pattern = r'\b[A-Z][a-zA-Z0-9\.]+(?:\s+[A-Z][a-zA-Z0-9\.]+){0,4}\s+(?:Pvt\.\s+Ltd\.|Ltd\.|Pvt\s+Ltd|Bank|Corporation|Techniques|Technologies|Assets)\b'
    matches = re.findall(pattern, full_text)
    ignore = ["BEFORE THE", "HON'BLE COURT", "DEBTS RECOVERY", "MEMO FILED", "FILED ON"]
    candidates = [m for m in matches if m.upper() not in ignore]
    return [entity for entity, count in Counter(candidates).most_common(15)]

# 4. THE STABLE FORENSIC JUDGE
def consult_gemini_forensic(image_crop, ai_guess, scan_guess, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        PIXEL-ONLY TRANSCRIPTION TASK:
        Transcribe the text in this image exactly.
        
        WARNING:
        The AI suggested: "{ai_guess}" (This may be a hallucination/guess).
        The OCR suggested: "{scan_guess}".
        
        STRICT RULES:
        1. If the text is faded/destroyed, output "[ILLEGIBLE]".
        2. Do NOT fix typos. Do NOT use generic names like "Innovative Systems" if not visible.
        3. Output ONLY the transcribed text.
        """
        
        for attempt in range(3):
            try:
                response = model.generate_content([prompt, image_crop])
                return response.text.strip()
            except Exception as e:
                if "429" in str(e): 
                    time.sleep(5 * (attempt + 1))
                    continue
                raise e
        return "[ILLEGIBLE]"
    except Exception as e:
        return f"[MANUAL_REVIEW_REQUIRED: {str(e)[:15]}]"

# 5. SCALED PIPELINE (Batch-Based)
def run_scaled_pipeline(file_path, lk, gk, live_view):
    logs = []
    
    # STEP 1: PARSING (LlamaParse handles large files better than raw OCR)
    live_view.info("🧠 Step 1: LlamaParse Deep Analysis...")
    parser = LlamaParse(api_key=lk, result_type="markdown")
    docs = parser.load_data(file_path)
    final_text = "\n\n".join([d.text for d in docs])
    
    # Build Knowledge Base
    golden_truths = extract_global_entities(final_text)
    logs.append(f"Knowledge Base: Found {len(golden_truths)} verified entities.")

    # STEP 2: BATCHED OCR MAPPING
    live_view.info("📸 Step 2: Batched Spatial Mapping (Memory Optimized)...")
    
    # Get page count without loading images
    info = pdfinfo_from_path(file_path)
    total_pages = info["Pages"]
    
    full_ocr_data = []
    # Process in 5-page batches to prevent RAM crashes
    batch_size = 5
    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        live_view.caption(f"Reading ink data: Pages {start}-{end} of {total_pages}...")
        
        batch_pages = convert_from_path(file_path, first_page=start, last_page=end, dpi=200)
        
        for pg_idx, pg in enumerate(batch_pages):
            data = pytesseract.image_to_data(pg, output_type=Output.DICT)
            for j, word in enumerate(data['text']):
                if len(word.strip()) > 2:
                    full_ocr_data.append({
                        "text": word, 
                        "box": (data['left'][j], data['top'][j], data['width'][j], data['height'][j]), 
                        "img": pg.copy() # Store copy, allow original to be garbage collected
                    })
        del batch_pages # Clear batch from RAM

    # STEP 3: CONTEXTUAL AUDIT
    live_view.info("⚖️ Step 3: Forensic Judicial Audit...")
    raw_blob = " ".join([x["text"] for x in full_ocr_data])
    ai_entities = set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', final_text))
    
    manual_needed = []
    
    for word in ai_entities:
        if word.upper() in ["BEFORE", "COURT", "INDIA", "STAMP", "DATE", "MEMO", "THE", "AND"]: continue
        
        # Conflict Detection
        if fuzz.partial_ratio(word.lower(), raw_blob.lower()) < 80:
            
            # A. Check Knowledge Base First (Cheapest & Most Reliable)
            match_truth, score = process.extractOne(word, golden_truths) if golden_truths else (None, 0)
            if score > 90:
                final_text = safe_replace_entity(final_text, word, match_truth)
                continue
            
            # B. Consult Forensic Judge
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
                
                if "[ILLEGIBLE]" in verdict or "[MANUAL" in verdict:
                    manual_needed.append({"word": word, "crop": crop, "page": best_match.get("page", 0)})
                else:
                    final_text = safe_replace_entity(final_text, word, verdict)
                    logs.append(f"Verified: {word} -> {verdict}")

    return final_text, manual_needed, logs

# 6. UI
tab1, tab2 = st.tabs(["🚀 Scaled Pipeline", "🔍 Manual Review"])

with st.sidebar:
    st.header("🔑 API Credentials")
    lk = st.text_input("LlamaCloud Key", type="password")
    gk = st.text_input("Gemini Key", type="password")
    
    if st.session_state.processed_data:
        st.divider()
        count = len(st.session_state.processed_data)
        if count == 1:
            name, text = list(st.session_state.processed_data.items())[0]
            st.download_button(f"📥 Download {name}.md", text, f"{name}.md")
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for n, c in st.session_state.processed_data.items(): zf.writestr(n+".md", c)
            st.download_button("� Download All (ZIP)", zip_buffer.getvalue(), "verified_docs.zip")

with tab1:
    st.subheader("Process Large Documents (100+ Pages)")
    files = st.file_uploader("Upload Legal Bundles (PDF)", accept_multiple_files=True)
    
    if st.button("� Run Scaled Analysis"):
        if not lk or not gk: st.error("Keys required!"); st.stop()
        
        live_view = st.empty()
        st.session_state.processed_data = {}
        st.session_state.review_queue = []
        
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                path = tmp.name
            
            res, manual, logs = run_scaled_pipeline(path, lk, gk, live_view=live_view)
            
            if manual:
                st.session_state.review_queue.append({"name": f.name, "text": res, "items": manual})
            else:
                st.session_state.processed_data[f.name] = res
            
        st.success("Batch Complete. Use the Review tab for illegible sections.")

with tab2:
    if st.session_state.review_queue:
        for f_idx, review in enumerate(st.session_state.review_queue):
            st.markdown(f"### Reviewing: {review['name']}")
            for i_idx, item in enumerate(review['items']):
                c1, c2 = st.columns([1, 3])
                with c1: st.image(item['crop'])
                with c2:
                    val = st.text_input(f"Transcribe word (Found: {item['word']})", key=f"r_{f_idx}_{i_idx}")
                    if st.button("Confirm", key=f"b_{f_idx}_{i_idx}"):
                        review['text'] = safe_replace_entity(review['text'], item['word'], val)
                        st.toast("Updated!")
            
            if st.button(f"✅ Finalize {review['name']}", key=f"fin_{f_idx}"):
                st.session_state.processed_data[review['name']] = review['text']
                st.session_state.review_queue.pop(f_idx)
                st.rerun()
    else:
        st.info("No items in the review queue.")
