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

# Custom CSS
st.markdown("""
    <style>
        .judge-card {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #FF4B4B;
            margin-bottom: 10px;
        }
        .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

if "processed_data" not in st.session_state: st.session_state.processed_data = {}

# 2. PROMPTS
PROMPT_LIBRARY = {
    "⚖️ Legal": "Transcribe text exactly. Use visual context to read faded names.",
    "🧩 Evidence": "Output Grids as Markdown Tables.",
}

# 3. SAFETY LOGIC
def is_bad_correction(original, proposal):
    if len(original) > 4 and len(proposal) < 3: return True
    if original.isalpha() and not proposal.replace(" ", "").isalpha(): return True
    return False

# 4. THE VISION JUDGE
def consult_gemini_vision(image_crop, option_a, option_b, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Read the text in this image crop EXACTLY.
        Context: Legal Document.
        Option A: "{option_a}"
        Option B: "{option_b}"
        Output ONLY the text you see.
        """
        response = model.generate_content([prompt, image_crop])
        return response.text.strip()
    except:
        return option_a

# 5. CORE PROCESSOR
def process_file(file_path, api_key_llama, api_key_google, live_view_container):
    logs = []
    with live_view_container.container():
        st.info("🧠 Reading document structure...")
    parser = LlamaParse(api_key=api_key_llama, result_type="markdown")
    docs = parser.load_data(file_path)
    ai_text = "\n\n".join([d.text for d in docs])
    final_text = ai_text

    with live_view_container.container():
        st.info("📸 Mapping word coordinates...")
    
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

    # JUDGMENT LOOP
    ignore = ["BEFORE", "THE", "AND", "BETWEEN", "DATE", "FILED", "MEMO", "HIGH", "COURT", "STAMP", "INDIA", "GOVERNMENT", "APPLICANT", "DEFENDANT", "COUNSEL", "ADVOCATE", "HYDERABAD", "TRIBUNAL", "ORDER"]
    
    ai_words = set(re.findall(r'\b[A-Z][a-z]{2,}\b', ai_text))
    ai_words.update(set(re.findall(r'\b[A-Z]{3,}\b', ai_text)))
    
    for word in ai_words:
        clean_word = word.strip(".,;:").upper()
        if clean_word in ignore: continue
        
        if fuzz.partial_ratio(word.lower(), raw_scan_text.lower()) < 85:
            best_match = None
            best_score = 0
            for item in full_ocr_data:
                score = fuzz.ratio(word.lower(), item["text"].lower())
                if score > best_score:
                    best_score = score
                    best_match = item
            
            if best_match:
                x, y, w, h = best_match["box"]
                crop = best_match["img"].crop((max(0, x-10), max(0, y-10), x+w+10, y+h+10))
                
                with live_view_container.container():
                    c1, c2 = st.columns([1, 2])
                    with c1: st.image(crop, caption="Evidence", width=150)
                    with c2:
                        st.markdown(f"**Conflict Detected:**")
                        st.code(f"AI:   {word}\nScan: {best_match['text']}")
                        with st.spinner("Gemini is judging..."):
                            verdict = consult_gemini_vision(crop, word, best_match["text"], api_key_google)
                        
                        if verdict != word:
                            if is_bad_correction(word, verdict):
                                st.warning(f"🛡️ Safety Block: Kept '{word}'")
                                logs.append(f"Safety: Kept '{word}'")
                            else:
                                st.markdown(f"✅ Verdict: **{verdict}**")
                                final_text = final_text.replace(word, verdict)
                                logs.append(f"Fixed: {word} -> {verdict}")
                        else:
                            st.caption("Judge upheld AI.")
                time.sleep(0.5)
    
    return final_text, logs

# 6. HELPER
def clean_final_output(text):
    if "[STAMP: CURRENT_PAGE_RAW_OCR_TEXT]" in text:
        text = text.split("[STAMP: CURRENT_PAGE_RAW_OCR_TEXT]")[0]
    return text.strip()

# 7. SMART DOWNLOAD FUNCTION
def render_download_button(location="sidebar"):
    if not st.session_state.processed_data:
        return

    count = len(st.session_state.processed_data)
    
    # CASE 1: SINGLE FILE -> Direct Markdown Download
    if count == 1:
        filename, content = list(st.session_state.processed_data.items())[0]
        clean_name = os.path.splitext(filename)[0] + ".md"
        
        if location == "sidebar":
            st.sidebar.download_button(
                label=f"📥 Download {clean_name}",
                data=content,
                file_name=clean_name,
                mime="text/markdown"
            )
        else:
            st.download_button(
                label=f"📥 DOWNLOAD {clean_name}",
                data=content,
                file_name=clean_name,
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )

    # CASE 2: MULTIPLE FILES -> Zip Download
    else:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for n, c in st.session_state.processed_data.items():
                zf.writestr(os.path.splitext(n)[0]+".md", c)
        
        if location == "sidebar":
            st.sidebar.download_button(
                label=f"📥 Download All ({count} files)",
                data=zip_buffer.getvalue(),
                file_name="verified_docs.zip",
                mime="application/zip"
            )
        else:
            st.download_button(
                label=f"📥 DOWNLOAD ALL ({count} FILES)",
                data=zip_buffer.getvalue(),
                file_name="verified_docs.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )

# 8. UI LAYOUT
with st.sidebar:
    st.header("🔑 Credentials")
    llama_key = st.text_input("LlamaCloud Key", type="password")
    google_key = st.text_input("Gemini Key", type="password")
    mode = st.selectbox("Scenario", list(PROMPT_LIBRARY.keys()))
    
    st.divider()
    render_download_button(location="sidebar")

st.title("👁️ Visual AI Judge")
st.markdown("### Real-Time Forensic Digitization")

files = st.file_uploader("Upload Files", accept_multiple_files=True)

if st.button("🚀 Start Live Session"):
    if not llama_key or not google_key: st.error("Keys required"); st.stop()
    if not files: st.warning("Upload a file"); st.stop()

    temp_dir = tempfile.mkdtemp()
    progress_bar = st.progress(0)
    
    st.divider()
    st.subheader("📺 Live Courtroom View")
    live_window = st.empty()
    
    # Reset data if new batch
    st.session_state.processed_data = {}
    
    for i, f in enumerate(files):
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as file: file.write(f.getbuffer())
        
        try:
            final_text, file_logs = process_file(path, llama_key, google_key, live_window)
            final_text = clean_final_output(final_text)
            st.session_state.processed_data[f.name] = final_text
            os.remove(path)
        except Exception as e:
            st.error(f"Error {f.name}: {e}")
            
        progress_bar.progress((i+1)/len(files))

    live_window.empty() 
    
    st.divider()
    st.success("✅ All Files Processed Successfully!")
    
    # Show Smart Download Button (Main Area)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        render_download_button(location="main")
        
    st.balloons()
