import streamlit as st
import os
import tempfile
import zipfile
import io
import nest_asyncio
import re
import pytesseract
from thefuzz import fuzz
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

st.set_page_config(page_title="Paranoid Legal Digitizer", page_icon="🛡️", layout="wide")

if not POPPLER_INSTALLED:
    st.error("Poppler not found. Install poppler-utils to enable PDF preview in Human Review.")

# Initialize State
if "review_queue" not in st.session_state: st.session_state.review_queue = []
if "processed_data" not in st.session_state: st.session_state.processed_data = {}

# 2. PROMPT LIBRARY (STRICT MODE)
PROMPT_LIBRARY = {
    "⚖️ Legal (Strict)": """
        You are a robotic legal transcription engine. 
        INSTRUCTION: Transcribe text letter-for-letter. 
        - DO NOT correct typos (e.g., if input says "Pegasvs", write "Pegasvs").
        - DO NOT expand abbreviations (e.g., do not change "Pvt Ltd" to "Private Limited").
        - DO NOT autocomplete names.
        - If text is unreadable, write [?].
        
        STRUCTURE:
        1. STAMPS: **[STAMP: <text>]**
        2. TABLES: Output as Markdown Tables.
    """,
    "🧩 Evidence (Jatakam)": """
        You are digitizing evidentiary grids.
        1. GRID: Output South/North Indian charts as Markdown Tables.
        2. SCRIPT: Keep Telugu/Sanskrit script exact.
        3. SPATIAL: Maintain box positions.
    """
}

# 3. THE LIE DETECTOR FUNCTION
def validate_accuracy(image_path, ai_text):
    """
    Compares AI output against Raw Tesseract OCR.
    Returns: (is_clean, error_message)
    """
    # 1. Get "Ground Truth" using Dumb OCR
    try:
        raw_text = pytesseract.image_to_string(image_path)
    except Exception:
        return False, "OCR Engine Failed"

    # 2. Extract "High Risk" Entities from AI Text (Capitalized Words > 4 chars)
    ignore_list = ["BEFORE", "THE", "AND", "BETWEEN", "DATE", "FILED", "MEMO", "HIGH", "COURT", "STAMP"]
    
    # Simple regex to find capitalized words (Potential Names/Companies)
    ai_words = set(re.findall(r'\b[A-Z][a-z]+\b', ai_text))
    # Add fully uppercase words too
    ai_words.update(re.findall(r'\b[A-Z]{4,}\b', ai_text))
    
    suspicious_words = []
    
    # 3. Cross-Examine
    for word in ai_words:
        if word.upper() in ignore_list:
            continue
            
        # Check if this word exists in the Raw Text (Fuzzy Match to allow small OCR typos)
        match_score = fuzz.partial_ratio(word.lower(), raw_text.lower())
        
        if match_score < 85:
            suspicious_words.append(word)

    if suspicious_words:
        return False, f"Hallucination Risk! AI found these words but Raw OCR did not: {', '.join(suspicious_words[:5])}"
    
    return True, "Verified"

# 4. MAIN APP
def main():
    with st.sidebar:
        st.header("🛡️ Paranoid Mode")
        api_key = st.text_input("LlamaCloud API Key", type="password")
        mode = st.selectbox("Scenario", list(PROMPT_LIBRARY.keys()))
        
        st.info("Validation Level: High")
        
        if st.session_state.processed_data:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for n, c in st.session_state.processed_data.items():
                    zf.writestr(os.path.splitext(n)[0]+".md", c)
            st.download_button("📥 Download Verified ZIP", zip_buffer.getvalue(), "verified.zip", "application/zip")

    st.title("🛡️ Zero-Hallucination Digitizer")

    tab1, tab2 = st.tabs(["Process", "Review"])

    with tab1:
        files = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Start"):
            if not api_key:
                st.error("Please provide a LlamaCloud API Key.")
            else:
                parser = LlamaParse(api_key=api_key, result_type="markdown", parsing_instruction=PROMPT_LIBRARY[mode])
                temp_dir = tempfile.mkdtemp()
                bar = st.progress(0)
                
                for i, f in enumerate(files):
                    path = os.path.join(temp_dir, f.name)
                    with open(path, "wb") as file: file.write(f.getbuffer())
                    
                    try:
                        # 1. Smart Parse
                        docs = parser.load_data(path)
                        ai_text = "\n".join([d.text for d in docs])
                        
                        # 2. Prepare Image for Validation (Page 1)
                        if f.name.lower().endswith(".pdf"):
                            images = convert_from_path(path, first_page=1, last_page=1)
                            val_image = images[0]
                        else:
                            val_image = path # Works for jpg/png
                        
                        # 3. RUN THE LIE DETECTOR
                        is_clean, msg = validate_accuracy(val_image, ai_text)
                        
                        # 4. Routing
                        if not is_clean or len(ai_text) < 50:
                            st.session_state.review_queue.append({
                                "name": f.name, "path": path, "text": ai_text, "error": msg
                            })
                        else:
                            st.session_state.processed_data[f.name] = ai_text
                            os.remove(path)
                            
                    except Exception as e:
                        st.error(f"Error processing {f.name}: {e}")
                    
                    bar.progress((i+1)/len(files))
                st.success("Done.")

    with tab2:
        if st.session_state.review_queue:
            q = st.session_state.review_queue
            sel = st.selectbox("Review File", [x["name"] for x in q])
            item = next(x for x in q if x["name"] == sel)
            
            st.error(f"⚠️ {item['error']}")
            c1, c2 = st.columns(2)
            with c1:
                # Display Image
                try:
                    if item["name"].lower().endswith(".pdf"):
                        st.image(convert_from_path(item["path"], first_page=1, last_page=1)[0], use_container_width=True)
                    else:
                        st.image(item["path"], use_container_width=True)
                except Exception as e:
                    st.error(f"Could not display image: {e}")
            with c2:
                new_text = st.text_area("Edit", item["text"], height=600)
                if st.button("Verify & Save"):
                    st.session_state.processed_data[sel] = new_text
                    st.session_state.review_queue = [x for x in q if x["name"] != sel]
                    st.rerun()
        else:
            st.success("Queue Empty")

if __name__ == "__main__":
    main()
