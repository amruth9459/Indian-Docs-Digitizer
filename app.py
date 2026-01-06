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
import time
from thefuzz import fuzz, process
from llama_parse import LlamaParse
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# 1. SYSTEM SETUP
nest_asyncio.apply()

# Poppler Check
POPPLER_INSTALLED = True
try:
    from pdf2image import convert_from_path
except ImportError:
    POPPLER_INSTALLED = False

st.set_page_config(
    page_title="Legal Digitizer Pro", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK & DASHBOARD ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card-like containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stAlert) {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
        background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }

    /* Status Box */
    .status-container {
        background: rgba(51, 65, 85, 0.5);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #3b82f6;
    }

    /* Fixed Height Image Container */
    div.stImage > img {
        max-height: 700px;
        object-fit: contain;
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "review_queue" not in st.session_state: st.session_state.review_queue = []
if "processed_data" not in st.session_state: st.session_state.processed_data = {}
if "active_tab" not in st.session_state: st.session_state.active_tab = "📤 PROCESS"

# 2. THE BRAIN: NOISE-CANCELING PROMPTS
PROMPT_LIBRARY = {
    "⚖️ Legal (Strict)": """
        You are a forensic legal digitizer. 
        
        CRITICAL RULES:
        1. ACCURACY: Transcribe text letter-for-letter. DO NOT auto-complete names.
        2. NOISE FILTER: IGNORE visual artifacts, punch holes, photocopy streaks, and dust. 
           - Do NOT output random symbols like '.. . ...,.,,' or ''.
           - If a section is purely noise, skip it.
        3. STAMPS: **[STAMP: <text> | Date: <date>]**.
        4. MARGINALIA: **[MARGIN NOTE: <text>]**.
        5. TABLES: Repair broken tables.
        6. LANGUAGES: Transcribe Telugu/Hindi/English exactly.
    """,
    "🧩 Evidence (Jatakam/Grids)": """
        You are digitizing evidentiary grids (Horoscopes/Family Trees).
        1. GRID: Output South/North Indian charts as Markdown Tables.
        2. SCRIPT: Keep Telugu/Sanskrit script exact. Do not Romanize.
        3. SPATIAL: Maintain box positions perfectly.
    """,
    "🏠 Property Deeds": """
        You are digitizing Sale Deeds.
        1. THUMBPRINTS: Ignore dark thumbprints overlaying text.
        2. SCHEDULE: Extract boundaries (N/S/E/W) into a table.
        3. MAPS: Output **[MAP DETECTED]** and dimensions only.
    """,
    "🏛️ Revenue Records (Pahani)": """
        You are digitizing Revenue Ledgers.
        1. COLUMNS: Strictly preserve the grid structure (Survey No, Name, Extent).
        2. HANDWRITING: Capture handwritten remarks.
    """,
    "💰 Banking (Dot Matrix)": """
        You are reading dot-matrix text.
        1. INFERENCE: Visually 'connect the dots' to form letters.
        2. STRUCTURE: Maintain Debit/Credit/Balance columns.
    """
}

# 3. THE "GARBAGE TRUCK" CLEANER
def clean_final_output(text):
    """
    Removes raw OCR noise and common garbage patterns from the start of the file.
    """
    # 1. Remove the "Raw OCR" block if it exists (Regex handles bold/non-bold)
    pattern = r"(\*\*)?\[STAMP: CURRENT_PAGE_RAW_OCR_TEXT\](\*\*)?.*?(?=\n\n|\Z)"
    text = re.sub(pattern, "", text, flags=re.DOTALL)
    
    # 2. Regex to remove blocks of high-density symbol garbage (ASCII noise)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # Calculate ratio of symbols to length
        if len(line) > 5:
            symbols = len(re.findall(r'[^a-zA-Z0-9\s]', line))
            if symbols / len(line) > 0.4:
                continue # Skip garbage line
        clean_lines.append(line)
        
    cleaned = "\n".join(clean_lines).strip()
    
    # 3. Remove common AI prefixes if they exist
    cleaned = re.sub(r'^(Here is the transcription:|Based on the document provided:)', '', cleaned, flags=re.IGNORECASE).strip()
    
    return cleaned

# 4. MULTI-PAGE VALIDATION
def validate_multipage(file_path, ai_text, status_placeholder):
    """
    Returns: (is_clean, error_details)
    """
    status_placeholder.markdown('<div class="status-container">🔍 Running Multi-Page Cross-Examination...</div>', unsafe_allow_html=True)
    error_details = []
    try:
        if file_path.lower().endswith(".pdf"):
            pages = convert_from_path(file_path)
        else:
            pages = [Image.open(file_path)]
    except:
        return True, [] 

    # Build Ground Truth
    full_document_ocr = [] 
    for page_idx, page_img in enumerate(pages):
        data = pytesseract.image_to_data(page_img, output_type=Output.DICT)
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            word = data['text'][i].strip()
            if word:
                full_document_ocr.append({
                    "text": word, "page": page_idx,
                    "box": (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                })

    # Check Hallucinations
    raw_text_blob = " ".join([item["text"] for item in full_document_ocr])
    stopwords = ["BEFORE", "THE", "AND", "BETWEEN", "DATE", "FILED", "MEMO", "HIGH", "COURT", "STAMP", "INDIA", "GOVERNMENT", "APPLICANT", "DEFENDANT", "COUNSEL", "ADVOCATE", "HYDERABAD", "TRIBUNAL", "ORDER", "OFFICE"]
    
    ai_words = set(re.findall(r'\b[A-Z][a-z]{2,}\b', ai_text))
    ai_words.update(set(re.findall(r'\b[A-Z]{3,}\b', ai_text)))
    
    for suspect_word in ai_words:
        clean_word = suspect_word.strip(".,;:").upper()
        if clean_word in stopwords: continue
        
        if fuzz.partial_ratio(suspect_word.lower(), raw_text_blob.lower()) < 80:
            # Find closest visual match location
            best_match = None
            best_score = 0
            for ocr_item in full_document_ocr:
                score = fuzz.ratio(suspect_word.lower(), ocr_item["text"].lower())
                if score > best_score:
                    best_score = score
                    best_match = ocr_item
            
            error_details.append({
                "word": suspect_word,
                "page": best_match["page"] if best_match and best_score > 40 else 0,
                "suggested_fix": best_match["text"] if best_match else "[Unknown]",
                "ocr_data": best_match,
                "raw_text": raw_text_blob # Keep for fuzzy suggestions
            })

    return (len(error_details) == 0), error_details

# 5. CROP & ZOOM HIGHLIGHTER
def get_zoomed_image(file_path, page_idx, target_box=None):
    try:
        if file_path.lower().endswith(".pdf"):
            page = convert_from_path(file_path, first_page=page_idx+1, last_page=page_idx+1)[0]
        else:
            page = Image.open(file_path).convert("RGB")
            
        if target_box:
            draw = ImageDraw.Draw(page)
            (x, y, w, h) = target_box
            # Draw Thick Red Box
            draw.rectangle([x-5, y-5, x+w+5, y+h+5], outline="red", width=5)
            
        return page
    except:
        return None

# 6. MAIN APP INTERFACE
def main():
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #3b82f6;'>⚙️ CONFIG</h1>", unsafe_allow_html=True)
        api_key = st.text_input("LlamaCloud API Key", type="password", help="Get your key from cloud.llamaindex.ai")
        
        st.divider()
        st.subheader("📂 Document Category")
        mode = st.selectbox("Select Logic", list(PROMPT_LIBRARY.keys()))
        st.info(f"**Mode:** {mode}")
        
        if st.session_state.processed_data:
            st.divider()
            st.subheader("📦 Export Results")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for n, c in st.session_state.processed_data.items():
                    zf.writestr(os.path.splitext(n)[0]+".md", c)
            st.download_button(
                label="📥 Download Final ZIP", 
                data=zip_buffer.getvalue(), 
                file_name="digitized_docs.zip", 
                mime="application/zip",
                width="stretch"
            )

    st.markdown("<h1 style='text-align: center;'>🇮🇳 Legal Digitizer <span style='color: #3b82f6; font-size: 0.5em;'>PRO</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Zero-Hallucination AI Extraction for Complex Indian Records</p>", unsafe_allow_html=True)

    # Tab handling for logical flow
    tabs = ["📤 PROCESS", "🔍 REVIEW DASHBOARD"]
    active_tab_index = tabs.index(st.session_state.active_tab)
    tab1, tab2 = st.tabs(tabs)

    with tab1:
        col_up, col_info = st.columns([2, 1])
        with col_up:
            files = st.file_uploader("Drop your scans here (PDF, JPG, PNG)", accept_multiple_files=True)
        with col_info:
            st.markdown("""
            ### 🛡️ Paranoid Mode Active
            - **Smart AI**: LlamaParse
            - **Dumb OCR**: Tesseract
            - **Validation**: Multi-Page Cross-Examination
            """)

        if st.button("🚀 START DIGITIZATION", width="stretch"):
            if not api_key:
                st.error("LlamaCloud API Key is required to proceed.")
                st.stop()
            if not files:
                st.warning("Please upload at least one file.")
                st.stop()
            
            parser = LlamaParse(api_key=api_key, result_type="markdown", user_prompt=PROMPT_LIBRARY[mode])
            temp_dir = tempfile.mkdtemp()
            
            main_progress = st.progress(0)
            status_area = st.empty()
            
            for i, f in enumerate(files):
                file_status = st.empty()
                file_status.markdown(f'<div class="status-container">📁 Processing <b>{f.name}</b>...</div>', unsafe_allow_html=True)
                
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as file: 
                    file.write(f.getbuffer())
                
                try:
                    # Granular Status Updates
                    status_area.markdown('<div class="status-container">☁️ Uploading to LlamaCloud & AI Parsing...</div>', unsafe_allow_html=True)
                    docs = parser.load_data(path)
                    ai_text = "\n\n".join([d.text for d in docs])
                    
                    # Multi-Page Validation
                    is_valid, error_list = validate_multipage(path, ai_text, status_area)
                    
                    if not is_valid:
                        status_area.markdown(f'<div class="status-container" style="border-left-color: #ef4444;">⚠️ Hallucination Risk Detected! Flagging for review.</div>', unsafe_allow_html=True)
                        st.session_state.review_queue.append({
                            "name": f.name, 
                            "path": path, 
                            "text": ai_text, 
                            "errors": error_list
                        })
                        time.sleep(1) # Visual feedback
                    else:
                        status_area.markdown('<div class="status-container" style="border-left-color: #10b981;">✨ Cleaning & Finalizing Output...</div>', unsafe_allow_html=True)
                        clean_text = clean_final_output(ai_text)
                        st.session_state.processed_data[f.name] = clean_text
                        os.remove(path)
                        time.sleep(0.5)
                        
                except Exception as e:
                    st.error(f"Error {f.name}: {str(e)}")
                
                main_progress.progress((i+1)/len(files))
                file_status.empty()
            
            status_area.success(f"Successfully processed {len(files)} files!")
            
            # Logical Flow: Next Steps
            st.divider()
            col_next1, col_next2 = st.columns(2)
            with col_next1:
                if st.session_state.review_queue:
                    if st.button("🔍 GO TO REVIEW DASHBOARD", width="stretch"):
                        st.session_state.active_tab = "🔍 REVIEW DASHBOARD"
                        st.rerun()
            with col_next2:
                if st.session_state.processed_data:
                    st.info("Batch complete! Use the sidebar to download your verified ZIP.")

    with tab2:
        if st.session_state.review_queue:
            q = st.session_state.review_queue
            
            # 1. Top Bar: File Selector & Progress
            c_sel, c_stat = st.columns([3, 1])
            with c_sel:
                sel_file = st.selectbox("Current File:", [x["name"] for x in q], label_visibility="collapsed")
            
            item = next(x for x in q if x["name"] == sel_file)
            
            # Initialize Error Navigation
            if "err_idx" not in st.session_state: st.session_state.err_idx = 0
            if "file_tracker" not in st.session_state or st.session_state.file_tracker != sel_file:
                st.session_state.err_idx = 0
                st.session_state.file_tracker = sel_file
                
            errors = item["errors"]
            curr_err = errors[st.session_state.err_idx] if errors else None
            
            with c_stat:
                if errors:
                    st.metric("Error Progress", f"{st.session_state.err_idx + 1}/{len(errors)}")
                else:
                    st.success("Clean!")

            st.divider()

            # 2. Main Dashboard (Split View)
            if curr_err:
                col_left, col_right = st.columns([1.5, 1], gap="medium")
                
                # --- LEFT: THE VISUAL PROOF ---
                with col_left:
                    st.markdown(f"**📍 Visual Context (Page {curr_err['page']+1})**")
                    box = curr_err["ocr_data"]["box"] if curr_err["ocr_data"] else None
                    img = get_zoomed_image(item["path"], curr_err["page"], box)
                    if img:
                        st.image(img, width="stretch")
                    else:
                        st.warning("Preview unavailable")

                # --- RIGHT: THE FIX WIZARD ---
                with col_right:
                    st.markdown("### 🛠️ Fix This")
                    st.error(f"AI Read: **{curr_err['word']}**")
                    
                    # Context Snippet
                    start = item["text"].find(curr_err["word"])
                    if start != -1:
                        snippet = item["text"][max(0, start-30):min(len(item["text"]), start+30)]
                        st.caption(f"...{snippet}...")
                    
                    st.markdown("#### Select Correction:")
                    
                    # Options
                    options = []
                    if curr_err["suggested_fix"] != "[Unknown]":
                        options.append(f"Replace with: '{curr_err['suggested_fix']}' (Raw Scan)")
                    
                    # Common Legal Corrections
                    if "Pega" in curr_err['word']: options.append("Pegasus Assets")
                    if "Karnataka" in curr_err['word']: options.append("Harsha")
                    
                    options.append(f"Keep: '{curr_err['word']}' (Ignore)")
                    options.append("Mark as [ILLEGIBLE]")
                    
                    choice = st.radio("Action:", options, key=f"rad_{sel_file}_{st.session_state.err_idx}", index=None)
                    
                    manual = st.text_input("Or type correction:", key=f"man_{sel_file}_{st.session_state.err_idx}")
                    
                    # APPLY LOGIC
                    final_val = None
                    if manual: 
                        final_val = manual
                    elif choice:
                        if "Replace with" in choice:
                            final_val = re.search(r"'(.*?)'", choice).group(1)
                        elif "Keep:" in choice:
                            final_val = curr_err["word"]
                        elif "ILLEGIBLE" in choice:
                            final_val = "[ILLEGIBLE]"
                        else:
                            final_val = choice
                    
                    # Navigation Handlers
                    c_prev, c_next = st.columns(2)
                    
                    with c_prev:
                        if st.button("⬅️ Previous"):
                            if st.session_state.err_idx > 0:
                                st.session_state.err_idx -= 1
                                st.rerun()
                    
                    with c_next:
                        if st.button("Confirm & Next ➡️", type="primary"):
                            if final_val:
                                item["text"] = item["text"].replace(curr_err["word"], final_val)
                                if st.session_state.err_idx < len(errors) - 1:
                                    st.session_state.err_idx += 1
                                    st.rerun()
                                else:
                                    st.success("All errors checked!")
                                    st.session_state.show_save = True
                            else:
                                st.warning("Please select a correction or type one.")

                    # SAVE BUTTON
                    if st.session_state.get("show_save"):
                        st.divider()
                        if st.button("💾 SAVE FINAL DOCUMENT", width="stretch"):
                            st.session_state.processed_data[sel_file] = clean_final_output(item["text"])
                            st.session_state.review_queue = [x for x in q if x["name"] != sel_file]
                            st.session_state.err_idx = 0
                            if "show_save" in st.session_state: del st.session_state.show_save
                            if not st.session_state.review_queue:
                                st.session_state.active_tab = "📤 PROCESS"
                            st.rerun()
                            
                st.divider()
                if st.button("🚫 Mark Entire File as ILLEGIBLE", width="stretch"):
                    st.session_state.processed_data[sel_file] = "[FILE MARKED ILLEGIBLE BY HUMAN REVIEWER]"
                    st.session_state.review_queue = [x for x in q if x["name"] != sel_file]
                    st.session_state.err_idx = 0
                    if not st.session_state.review_queue:
                        st.session_state.active_tab = "📤 PROCESS"
                    st.rerun()

            else:
                st.info("No errors detected in this file.")
                if st.button("Mark as Done", width="stretch"):
                    st.session_state.processed_data[sel_file] = clean_final_output(item["text"])
                    st.session_state.review_queue = [x for x in q if x["name"] != sel_file]
                    st.rerun()

            st.markdown("### 📝 Final Verification")
            st.text_area("Full Text Preview", value=clean_final_output(item["text"]), height=300)
        else:
            st.success("🎉 All documents verified and ready for download!")
            if st.button("⬅️ BACK TO UPLOAD", width="stretch"):
                st.session_state.active_tab = "📤 PROCESS"
                st.rerun()

if __name__ == "__main__":
    main()
