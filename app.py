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

# 1. SYSTEM SETUP
nest_asyncio.apply()
st.set_page_config(page_title="Visual Judge Pro", page_icon="👁️", layout="wide")

# Custom CSS for a better UI
st.markdown("""
    <style>
        .stProgress > div > div > div > div { background-color: #4CAF50; }
        .status-text { font-family: monospace; font-size: 14px; color: #FFA500; }
        .success-box { padding: 10px; border-left: 5px solid #4CAF50; background: rgba(76, 175, 80, 0.1); }
        .log-container {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            font-size: 0.85em;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

if "processed_data" not in st.session_state: st.session_state.processed_data = {}

# 2. PROMPTS
PROMPT_LIBRARY = {
    "⚖️ Legal": "Transcribe text exactly. Use visual context to read faded names.",
    "🧩 Evidence": "Output Grids as Markdown Tables.",
}

# 3. API HEALTH CHECK
def check_api_keys(llama, google):
    errors = []
    if not llama or not llama.startswith("llx-"):
        errors.append("❌ Invalid LlamaCloud Key format.")
    if not google:
        errors.append("❌ Missing Google Gemini Key.")
    
    return errors

# 4. SAFETY LOGIC
def is_bad_correction(original, proposal):
    # Don't replace long words with tiny ones
    if len(original) > 4 and len(proposal) < 3: return True
    # Don't replace words with symbols
    if original.isalpha() and not proposal.replace(" ", "").isalpha(): return True
    return False

# 5. THE VISION JUDGE
def consult_gemini_vision(image_crop, option_a, option_b, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Expert Task: Read the text in this image crop EXACTLY.
        Context: It is a Legal Document.
        Contestants: "{option_a}" vs "{option_b}".
        Output ONLY the text visible in the crop. No explanations.
        """
        response = model.generate_content([prompt, image_crop])
        return response.text.strip()
    except Exception as e:
        return option_a # Fail safe

# 6. CORE PROCESSOR WITH DYNAMIC STATUS
def process_file(file_path, api_key_llama, api_key_google, status_callback):
    logs = []
    
    # STEP 1: PARSE (LlamaParse)
    status_callback("🧠 AI Reading Document...")
    parser = LlamaParse(api_key=api_key_llama, result_type="markdown")
    docs = parser.load_data(file_path)
    ai_text = "\n\n".join([d.text for d in docs])
    final_text = ai_text

    # STEP 2: OCR MAPPING (Tesseract)
    status_callback("📸 Scanning Coordinates...")
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
                if word:
                    full_ocr_data.append({
                        "text": word, "page": pg_idx,
                        "box": (data['left'][i], data['top'][i], data['width'][i], data['height'][i]),
                        "img": pg
                    })
        
        raw_scan_text = " ".join([x["text"] for x in full_ocr_data])
    except Exception as e:
        return ai_text, [f"OCR Error: {e}"]

    # STEP 3: CONFLICT DETECTION
    status_callback("🔎 Detecting Conflicts...")
    ignore = ["BEFORE", "THE", "AND", "BETWEEN", "DATE", "FILED", "MEMO", "HIGH", "COURT", "STAMP", "INDIA", "GOVERNMENT", "APPLICANT", "DEFENDANT", "COUNSEL", "ADVOCATE", "HYDERABAD", "TRIBUNAL", "ORDER"]
    
    ai_words = set(re.findall(r'\b[A-Z][a-z]{2,}\b', ai_text))
    ai_words.update(set(re.findall(r'\b[A-Z]{3,}\b', ai_text)))
    
    total_checks = len(ai_words)
    checked_count = 0

    for word in ai_words:
        checked_count += 1
        # Update status every 5 words so it doesn't flicker too fast
        if checked_count % 5 == 0:
            status_callback(f"⚖️ Judging Word {checked_count}/{total_checks}...")

        clean_word = word.strip(".,;:").upper()
        if clean_word in ignore: continue
        
        if fuzz.partial_ratio(word.lower(), raw_scan_text.lower()) < 85:
            
            # Find Match
            best_match = None
            best_score = 0
            for item in full_ocr_data:
                score = fuzz.ratio(word.lower(), item["text"].lower())
                if score > best_score:
                    best_score = score
                    best_match = item
            
            if best_match:
                # CROP & JUDGE
                x, y, w, h = best_match["box"]
                crop = best_match["img"].crop((max(0, x-10), max(0, y-10), x+w+10, y+h+10))
                
                verdict = consult_gemini_vision(crop, word, best_match["text"], api_key_google)
                
                if verdict != word:
                    if is_bad_correction(word, verdict):
                        logs.append(f"🛡️ SAFETY: Kept '{word}' (Judge said '{verdict}')")
                    else:
                        final_text = final_text.replace(word, verdict)
                        logs.append(f"👁️ FIXED: '{word}' -> '{verdict}'")
    
    return final_text, logs

# 7. HELPER
def clean_final_output(text):
    if "[STAMP: CURRENT_PAGE_RAW_OCR_TEXT]" in text:
        text = text.split("[STAMP: CURRENT_PAGE_RAW_OCR_TEXT]")[0]
    return text.strip()

# 8. UI LAYOUT
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Password inputs for security
    llama_key = st.text_input("LlamaCloud Key", type="password")
    google_key = st.text_input("Google Gemini Key", type="password")
    mode = st.selectbox("Scenario", list(PROMPT_LIBRARY.keys()))
    
    # Check button
    if llama_key and google_key:
        errs = check_api_keys(llama_key, google_key)
        if not errs:
            st.success("✅ Keys Validated")
        else:
            for e in errs: st.error(e)

    st.divider()
    
    if st.session_state.processed_data:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for n, c in st.session_state.processed_data.items():
                zf.writestr(os.path.splitext(n)[0]+".md", c)
        st.download_button("📥 Download Final ZIP", zip_buffer.getvalue(), "final_docs.zip", "application/zip", type="primary", width="stretch")

st.title("👁️ Visual AI Judge")
st.markdown("Automated forensic digitization pipeline with multi-modal verification.")

files = st.file_uploader("Upload Scans (PDF/Images)", accept_multiple_files=True)

if st.button("🚀 Start Pipeline", width="stretch"):
    # 1. PRE-FLIGHT CHECK
    errors = check_api_keys(llama_key, google_key)
    if errors:
        for e in errors: st.error(e)
        st.stop()
        
    if not files:
        st.warning("Please upload a file first.")
        st.stop()

    # 2. PROCESSING LOOP
    temp_dir = tempfile.mkdtemp()
    
    # Create containers for dynamic updates
    main_progress_bar = st.progress(0)
    current_action = st.empty()
    log_area = st.expander("Show Live Logs", expanded=True)
    
    for i, f in enumerate(files):
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as file: file.write(f.getbuffer())
        
        # Define a status update function to pass down
        def update_status(msg):
            current_action.markdown(f"**🔄 Processing {f.name}:** `{msg}`")
        
        try:
            # RUN PIPELINE
            final_text, file_logs = process_file(path, llama_key, google_key, update_status)
            final_text = clean_final_output(final_text)
            
            st.session_state.processed_data[f.name] = final_text
            
            # Print logs to UI in real-time
            with log_area:
                st.markdown(f"**{f.name}:**")
                for log in file_logs:
                    if "FIXED" in log: st.markdown(f":green[{log}]")
                    elif "SAFETY" in log: st.markdown(f":orange[{log}]")
                    else: st.text(log)
            
            os.remove(path)
            
        except Exception as e:
            st.error(f"Failed on {f.name}: {e}")
            
        # Update main bar
        main_progress_bar.progress((i+1)/len(files))

    current_action.success("✅ All Files Processed Successfully!")
    st.balloons()
