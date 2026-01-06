import streamlit as st
import os
import tempfile
import zipfile
import io
import nest_asyncio
import shutil
import re
import pytesseract
from pytesseract import Output
from thefuzz import fuzz, process
from llama_parse import LlamaParse
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
import time
from collections import Counter

# 1. SYSTEM SETUP
nest_asyncio.apply()
st.set_page_config(page_title="Contextual Forensic Digitizer", page_icon="⚖️", layout="wide")

# CSS
st.markdown("""
    <style>
        .judge-card { border-left: 5px solid #FF4B4B; background: #262730; padding: 10px; }
        .stButton>button { font-weight: bold; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

if "processed_data" not in st.session_state: st.session_state.processed_data = {}

# 2. PROMPTS
PROMPT_LIBRARY = {
    "⚖️ Legal": "Transcribe text exactly. Use visual context to read faded names.",
    "🧩 Evidence": "Output Grids as Markdown Tables.",
}

# 3. HELPER: CONTENT FILTER
def is_valid_content(word):
    if not any(c.isalnum() for c in word): return False
    if len(word) == 1 and word not in ['a', 'A', 'I', '&']: return False
    if len(word) > 20 and " " not in word: return False
    return True

# 4. HELPER: BAD CORRECTION CHECK
def is_bad_correction(original, proposal):
    if len(original) > 4 and len(proposal) < 3: return True
    if original.isalpha() and not proposal.replace(" ", "").isalpha(): return True
    return False

# 5. NEW FEATURE: GLOBAL CONTEXT EXTRACTION
def extract_global_entities(full_text):
    """
    Finds the 'Golden Truths' in the document.
    Returns a list of high-confidence entities (e.g. 'Pegasus Assets', 'Instruments Techniques')
    """
    # Regex to find Capitalized Phrases (e.g. "Instruments Techniques Pvt Ltd")
    # Matches words starting with Caps, allowing for abbreviations like "Pvt."
    pattern = r'\b[A-Z][a-zA-Z0-9\.]+(?:\s+[A-Z][a-zA-Z0-9\.]+){1,}\b'
    
    matches = re.findall(pattern, full_text)
    
    # Filter out common headers
    ignore = ["BEFORE THE", "HON'BLE COURT", "DEBTS RECOVERY", "MEMO FILED", "FOR THE", "AND THE", "IN THE MATTER", "FILED ON", "FILED BY"]
    candidates = [m for m in matches if m.upper() not in ignore and len(m) > 5]
    
    # Count frequency. If it appears more than once, it's likely a Truth.
    counts = Counter(candidates)
    golden_truths = [entity for entity, count in counts.items()]
    
    return golden_truths

# 6. VISION JUDGE
def consult_gemini_vision(image_crop, ai_guess, scan_guess, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are a Forensic Document Examiner.
        Task: Transcribe the text in this image crop EXACTLY.
        Context:
        - Input A (AI): "{ai_guess}"
        - Input B (Scan): "{scan_guess}"
        RULES:
        1. If image shows broken text like "Pegasu", write "Pegasu".
        2. Do NOT expand abbreviations.
        3. If unreadable, output "[ILLEGIBLE]".
        Output ONLY the visible text.
        """
        response = model.generate_content([prompt, image_crop])
        return response.text.strip()
    except:
        return scan_guess

# 7. CORE PROCESSOR
def process_file(file_path, api_key_llama, api_key_google, live_view_container):
    logs = []
    
    with live_view_container.container():
        st.info("🧠 Parsing Document...")
    
    # Parse
    parser = LlamaParse(api_key=api_key_llama, result_type="markdown")
    docs = parser.load_data(file_path)
    ai_text = "\n\n".join([d.text for d in docs])
    final_text = ai_text

    # --- STEP 1: BUILD KNOWLEDGE BASE ---
    with live_view_container.container():
        st.info("📚 Building Document Knowledge Base...")
    
    golden_truths = extract_global_entities(ai_text)
    logs.append(f"Found {len(golden_truths)} Context Entities (e.g. '{golden_truths[0] if golden_truths else 'None'}')")

    # --- STEP 2: OCR MAPPING ---
    try:
        if file_path.lower().endswith(".pdf"):
            pages = convert_from_path(file_path, dpi=300)
        else:
            pages = [Image.open(file_path)]
            
        full_ocr_data = []
        for pg_idx, pg in enumerate(pages):
            data = pytesseract.image_to_data(pg, output_type=Output.DICT)
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                if is_valid_content(word):
                    full_ocr_data.append({
                        "text": word, "page": pg_idx,
                        "box": (data['left'][i], data['top'][i], data['width'][i], data['height'][i]),
                        "img": pg
                    })
        raw_scan_text = " ".join([x["text"] for x in full_ocr_data])
    except Exception as e:
        return ai_text, [f"OCR Error: {e}"]

    # --- STEP 3: JUDGMENT LOOP ---
    ignore = ["BEFORE", "THE", "AND", "BETWEEN", "DATE", "FILED", "MEMO", "HIGH", "COURT", "STAMP", "INDIA", "GOVERNMENT", "APPLICANT", "DEFENDANT", "COUNSEL", "ADVOCATE", "HYDERABAD", "TRIBUNAL", "ORDER"]
    
    ai_words = set(re.findall(r'\b[A-Z][a-z]{2,}\b', ai_text))
    ai_words.update(set(re.findall(r'\b[A-Z]{3,}\b', ai_text)))
    
    for word in ai_words:
        clean_word = word.strip(".,;:").upper()
        if clean_word in ignore: continue
        if not is_valid_content(clean_word): continue
        
        # Conflict?
        if fuzz.partial_ratio(word.lower(), raw_scan_text.lower()) < 85:
            
            # --- NEW: GLOBAL CONTEXT CHECK (The Shortcut) ---
            # Before calling Gemini, check if this messy word matches a Golden Truth
            context_match = process.extractOne(word, golden_truths)
            if context_match and context_match[1] > 85:
                # Example: AI saw "Pegasu_s", Context found "Pegasus Assets" (90% match)
                # We trust the Context!
                final_text = final_text.replace(word, context_match[0])
                logs.append(f"📚 CONTEXT FIXED: '{word}' -> '{context_match[0]}' (Found elsewhere in doc)")
                continue # Skip Vision Judge, we solved it!

            # --- IF NO CONTEXT MATCH, CALL VISION JUDGE ---
            best_match = None
            best_score = 0
            for item in full_ocr_data:
                score = fuzz.ratio(word.lower(), item["text"].lower())
                if score > best_score:
                    best_score = score
                    best_match = item
            
            if best_match:
                x, y, w, h = best_match["box"]
                crop = best_match["img"].crop((max(0, x-5), max(0, y-5), x+w+5, y+h+5))
                
                with live_view_container.container():
                    c1, c2 = st.columns([1, 2])
                    with c1: st.image(crop, caption="Evidence", width=120)
                    with c2:
                        st.markdown(f"**Investigating:** `{word}`")
                        
                        with st.spinner("Forensic analysis..."):
                            verdict = consult_gemini_vision(crop, word, best_match["text"], api_key_google)
                        
                        if verdict != word:
                            if is_bad_correction(word, verdict):
                                logs.append(f"🛡️ SAFETY: Kept '{word}'")
                            else:
                                final_text = final_text.replace(word, verdict)
                                st.success(f"Correction: {verdict}")
                                logs.append(f"👁️ VISION FIXED: {word} -> {verdict}")
                        else:
                            st.caption("AI upheld.")
                time.sleep(0.1)
    
    return final_text, logs

# 8. CLEANER
def clean_final_output(text):
    if "[STAMP: CURRENT_PAGE_RAW_OCR_TEXT]" in text:
        text = text.split("[STAMP: CURRENT_PAGE_RAW_OCR_TEXT]")[0]
    return text.strip()

# 9. UI LAYOUT
def render_download_button(location="sidebar"):
    if not st.session_state.processed_data: return
    count = len(st.session_state.processed_data)
    
    if count == 1:
        filename, content = list(st.session_state.processed_data.items())[0]
        clean_name = os.path.splitext(filename)[0] + ".md"
        btn_label = f"📥 Download {clean_name}"
        mime_type = "text/markdown"
        data = content
    else:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for n, c in st.session_state.processed_data.items(): zf.writestr(os.path.splitext(n)[0]+".md", c)
        btn_label = f"📥 Download All ({count} Files)"
        mime_type = "application/zip"
        data = zip_buffer.getvalue()
        clean_name = "verified_docs.zip"

    if location == "sidebar":
        st.sidebar.download_button(btn_label, data, clean_name, mime_type)
    else:
        st.download_button(btn_label, data, clean_name, mime_type, type="primary", use_container_width=True)

with st.sidebar:
    st.header("🔑 Credentials")
    llama_key = st.text_input("LlamaCloud Key", type="password")
    google_key = st.text_input("Gemini Key", type="password")
    mode = st.selectbox("Scenario", list(PROMPT_LIBRARY.keys()))
    st.divider()
    render_download_button("sidebar")

st.title("👁️ Contextual Forensic Digitizer")
st.markdown("### Uses full-document context to repair localized errors.")

files = st.file_uploader("Upload Files", accept_multiple_files=True)

if st.button("🚀 Start Contextual Analysis"):
    if not llama_key or not google_key: st.error("Keys required"); st.stop()
    if not files: st.warning("Upload a file"); st.stop()

    temp_dir = tempfile.mkdtemp()
    progress_bar = st.progress(0)
    
    st.divider()
    live_window = st.empty()
    st.session_state.processed_data = {}
    
    for i, f in enumerate(files):
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as file: file.write(f.getbuffer())
        
        try:
            final_text, file_logs = process_file(path, llama_key, google_key, live_window)
            final_text = clean_final_output(final_text)
            st.session_state.processed_data[f.name] = final_text
            os.remove(path)
            
            with st.expander(f"📜 Log: {f.name}", expanded=False):
                for log in file_logs:
                    if "CONTEXT" in log: st.markdown(f":blue[{log}]")
                    elif "VISION" in log: st.markdown(f":green[{log}]")
                    else: st.text(log)
                    
        except Exception as e:
            st.error(f"Error {f.name}: {e}")
            
        progress_bar.progress((i+1)/len(files))

    live_window.empty()
    st.divider()
    st.success("✅ Analysis Complete")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        render_download_button("main")
    st.balloons()
