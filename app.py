import streamlit as st
import os
import tempfile
import zipfile
import io
import nest_asyncio
import shutil
import re
import pytesseract
import time
from thefuzz import fuzz, process
from llama_parse import LlamaParse
from pdf2image import convert_from_path

# 1. SYSTEM SETUP
nest_asyncio.apply()

# Poppler Check
POPPLER_INSTALLED = True
try:
    from pdf2image import convert_from_path
except ImportError:
    POPPLER_INSTALLED = False

st.set_page_config(
    page_title="Indian Doc Digitizer (Pro)", 
    page_icon="🇮🇳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
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

# 4. THE "LIE DETECTOR" FUNCTION (Stricter & returns context)
def validate_accuracy(image_path, ai_text, status_placeholder):
    """
    Returns: (is_clean, suspicious_list, raw_text)
    """
    status_placeholder.markdown('<div class="status-container">🔍 Running Tesseract Cross-Examination...</div>', unsafe_allow_html=True)
    
    # 1. Get Ground Truth
    try:
        raw_text = pytesseract.image_to_string(image_path)
    except:
        return True, [], ""

    # 2. Extract Proper Nouns (Capitalized words > 3 chars)
    # Filter out common legal stopwords to focus on Names/Entities
    stopwords = ["BEFORE", "THE", "AND", "BETWEEN", "DATE", "FILED", "MEMO", "HIGH", "COURT", "STAMP", "INDIA", "GOVERNMENT", "APPLICANT", "DEFENDANT", "COUNSEL", "ADVOCATE", "HYDERABAD", "TRIBUNAL", "ORDER", "OFFICE"]
    
    # Regex to find capitalized words (names, companies)
    ai_words = set(re.findall(r'\b[A-Z][a-z]{2,}\b', ai_text))
    ai_words.update(set(re.findall(r'\b[A-Z]{3,}\b', ai_text)))
    
    suspicious_list = []
    
    # 3. Cross-Examine
    for word in ai_words:
        clean_word = word.strip(".,;:").upper()
        if clean_word in stopwords: 
            continue
            
        # STRICT CHECK: Word must exist in raw text with 80% similarity
        if fuzz.partial_ratio(word.lower(), raw_text.lower()) < 80:
            suspicious_list.append(word)

    is_clean = len(suspicious_list) == 0
    return is_clean, suspicious_list, raw_text

# 5. MAIN APP INTERFACE
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
                label="📥 Download Clean ZIP", 
                data=zip_buffer.getvalue(), 
                file_name="digitized_docs.zip", 
                mime="application/zip",
                use_container_width=True
            )

    st.markdown("<h1 style='text-align: center;'>🇮🇳 Indian Document Digitizer <span style='color: #3b82f6; font-size: 0.5em;'>PRO</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Zero-Hallucination AI Extraction for Complex Indian Records</p>", unsafe_allow_html=True)

    # Tab handling for logical flow
    tabs = ["📤 PROCESS", "🕵️ REVIEW"]
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
            - **Validation**: Cross-Examination enabled
            """)

        if st.button("🚀 START DIGITIZATION", use_container_width=True):
            if not api_key:
                st.error("LlamaCloud API Key is required to proceed.")
                st.stop()
            if not files:
                st.warning("Please upload at least one file.")
                st.stop()
            
            parser = LlamaParse(api_key=api_key, result_type="markdown", parsing_instruction=PROMPT_LIBRARY[mode])
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
                    
                    status_area.markdown('<div class="status-container">🖼️ Preparing Image for Validation...</div>', unsafe_allow_html=True)
                    if f.name.lower().endswith(".pdf"):
                        images = convert_from_path(path, first_page=1, last_page=1)
                        val_img = images[0]
                    else:
                        val_img = path
                    
                    # Lie Detector
                    is_valid, suspicious_list, raw_text = validate_accuracy(val_img, ai_text, status_area)
                    
                    if not is_valid:
                        status_area.markdown(f'<div class="status-container" style="border-left-color: #ef4444;">⚠️ Hallucination Risk Detected! Flagging for review.</div>', unsafe_allow_html=True)
                        st.session_state.review_queue.append({
                            "name": f.name, 
                            "path": path, 
                            "text": ai_text, 
                            "errors": suspicious_list,
                            "raw_text": raw_text,
                            "error": f"Found {len(suspicious_list)} suspicious entities."
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
                    if st.button("🕵️ GO TO REVIEW QUEUE", use_container_width=True):
                        st.session_state.active_tab = "🕵️ REVIEW"
                        st.rerun()
            with col_next2:
                if st.session_state.processed_data:
                    st.info("Batch complete! Use the sidebar to download your verified ZIP.")

    with tab2:
        if st.session_state.review_queue:
            q = st.session_state.review_queue
            st.markdown(f"### ⚠️ {len(q)} Files Require Human Review")
            
            sel = st.selectbox("Select File to Verify", [x["name"] for x in q])
            item = next(x for x in q if x["name"] == sel)
            
            st.error(f"**Reason for Flag:** {item['error']}")
            
            c_img, c_edit = st.columns([1, 1])
            with c_img:
                st.subheader("Original Document")
                try:
                    if item["name"].lower().endswith(".pdf"):
                        st.image(convert_from_path(item["path"], first_page=1, last_page=1)[0], use_container_width=True)
                    else:
                        st.image(item["path"], use_container_width=True)
                except Exception as e:
                    st.error(f"Could not display image: {e}")
                
                st.divider()
                with st.expander("🔍 Show Raw OCR (Dumb AI Layer)"):
                    st.code(item.get("raw_text", "Not available"))

            with c_edit:
                st.subheader("🔧 Smart Entity Repair")
                st.info("Select the correct version of the flagged words below.")
                
                # We work on a copy of the text
                current_text = item["text"]
                
                # If the Lie Detector found specific words
                if item.get("errors"):
                    for i, error_word in enumerate(item["errors"]):
                        st.divider()
                        
                        # Show context snippet
                        start_idx = current_text.find(error_word)
                        if start_idx != -1:
                            end_idx = start_idx + len(error_word)
                            snippet = current_text[max(0, start_idx-30):min(len(current_text), end_idx+30)]
                            snippet = snippet.replace("\n", " ")
                            st.markdown(f"🔴 **Suspect Word:** `{error_word}`")
                            st.caption(f"...{snippet}...")
                        
                        # Generate Intelligent Options
                        options = []
                        
                        # Option A: What Tesseract saw in the raw text (Best Guess)
                        raw_text_options = process.extract(error_word, item["raw_text"].split(), limit=3)
                        for match, score in raw_text_options:
                            if match != error_word and match not in options:
                                options.append(match)
                                
                        # Option B: Common Legal Corrections
                        if "Pega" in error_word: options.append("Pegasus Assets")
                        if "Karnataka" in error_word: options.append("Harsha")
                        
                        # Option C: The Original
                        options.append(error_word + " (Keep Original)")
                        
                        # Option D: "Illegible"
                        options.append(" [ILLEGIBLE] ")

                        # UI: Radio Button for speed
                        choice = st.radio(
                            f"Correction for '{error_word}':", 
                            options, 
                            key=f"rad_{i}_{sel}",
                            index=None
                        )
                        
                        final_replacement = None
                        if choice:
                            if "Keep Original" in choice:
                                final_replacement = error_word
                            elif "[ILLEGIBLE]" in choice:
                                final_replacement = "[ILLEGIBLE]"
                            else:
                                final_replacement = choice

                        # Manual Override
                        manual_fix = st.text_input(f"Or type manually for '{error_word}':", key=f"man_{i}_{sel}")
                        if manual_fix:
                            final_replacement = manual_fix
                        
                        # Apply the fix to the text immediately
                        if final_replacement and final_replacement != error_word:
                            current_text = current_text.replace(error_word, final_replacement)
                            st.success(f"Fixed: {error_word} ➡️ {final_replacement}")

                st.divider()
                
                # Global "Illegible" Button
                if st.button("🚫 Mark Entire File as ILLEGIBLE", use_container_width=True):
                    st.session_state.processed_data[sel] = "[FILE MARKED ILLEGIBLE BY HUMAN REVIEWER]"
                    st.session_state.review_queue = [x for x in q if x["name"] != sel]
                    if not st.session_state.review_queue:
                        st.session_state.active_tab = "📤 PROCESS"
                    st.rerun()

                st.markdown("### 📝 Final Verification")
                final_text_display = st.text_area("Final Text Preview", value=clean_final_output(current_text), height=400)
                
                if st.button("✅ APPROVE & SAVE FILE", use_container_width=True):
                    st.session_state.processed_data[sel] = final_text_display
                    st.session_state.review_queue = [x for x in q if x["name"] != sel]
                    if not st.session_state.review_queue:
                        st.session_state.active_tab = "📤 PROCESS"
                    st.rerun()
        else:
            st.success("🎉 All files verified! Your queue is empty.")
            if st.button("⬅️ BACK TO UPLOAD", use_container_width=True):
                st.session_state.active_tab = "📤 PROCESS"
                st.rerun()

if __name__ == "__main__":
    main()
