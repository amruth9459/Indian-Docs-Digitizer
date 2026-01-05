import streamlit as st
import os
import tempfile
import zipfile
import io
import nest_asyncio
import shutil
from pdf2image import convert_from_path
from llama_parse import LlamaParse
from PIL import Image

# Apply nest_asyncio for LlamaParse
nest_asyncio.apply()

# Poppler Check
POPPLER_INSTALLED = True
try:
    from pdf2image import convert_from_path
except ImportError:
    POPPLER_INSTALLED = False

# Page Config
st.set_page_config(page_title="Indian Doc Digitizer", layout="wide")

if not POPPLER_INSTALLED:
    st.error("Poppler not found. Install poppler-utils to enable PDF preview in Human Review.")


# Session State Initialization
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = {}

# Prompt Library
PROMPT_LIBRARY = {
    "⚖️ Legal (Court Orders/Notices)": """
        You are a forensic legal digitizer.
        1. STAMPS: **[STAMP: <text> | Date: <date>]**.
        2. MARGINALIA: **[MARGIN NOTE: <text>]**.
        3. TABLES: Repair broken tables spanning pages.
        4. LANGUAGES: Transcribe Telugu/Hindi/English exactly. DO NOT TRANSLATE.
        5. ARTIFACTS: Ignore punch holes, fax headers.
    """,
    "🔮 Astrology (Jatakam/Panchangam)": """
        You are an expert in Indian Astrology charts.
        1. GRIDS: Identify South Indian (Square) or North Indian (Diamond) charts. Represent them as Markdown Tables.
        2. PLANETS: Maintain the exact box position of planets (Grahas).
        3. TERMINOLOGY: Keep terms like 'Lagna', 'Rasi', 'Navamsa'.
        4. SCRIPT: Transcribe Telugu/Sanskrit script exactly. Do not Romanize.
    """,
    "🏠 Property Deeds (Sale/Registration)": """
        You are digitizing Registration Documents.
        1. THUMB IMPRESSIONS: Ignore purple/black thumbprints overlaying text. Read the name underneath.
        2. SCHEDULE: Extract property boundaries (North/South/East/West) into a table.
        3. MAPS: If a sketch map exists, output **[MAP DETECTED]** and extract written dimensions only.
        4. STAMPS: Transcribe Non-Judicial Stamp values.
    """,
    "🏛️ Revenue Records (Pahani/Adangal)": """
        You are digitizing Government Revenue Records.
        1. COLUMNS: Strictly preserve the grid structure of Survey No, Pattadar Name, Extent.
        2. HANDWRITING: Capture handwritten officer remarks in the 'Remarks' column.
    """,
    "💰 Banking (Dot Matrix Passbooks)": """
        You are reading dot-matrix printed text.
        1. INFERENCE: 'Connect the dots' to form letters (e.g., correct '5' to 'S' based on context).
        2. STRUCTURE: Maintain Debit/Credit/Balance columns.
    """
}

def main():
    st.title("🇮🇳 Indian Document Digitizer")
    
    # Sidebar
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("LlamaCloud API Key", type="password")
    selected_mode = st.sidebar.selectbox("Select Document Type", list(PROMPT_LIBRARY.keys()))
    
    if st.sidebar.button("Download Processed Data (ZIP)"):
        if st.session_state.processed_data:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for filename, content in st.session_state.processed_data.items():
                    zf.writestr(f"{filename}.md", content)
            st.sidebar.download_button(
                label="Click to Download ZIP",
                data=zip_buffer.getvalue(),
                file_name="digitized_docs.zip",
                mime="application/zip"
            )
        else:
            st.sidebar.warning("No processed data to download.")

    # Main UI Tabs
    tab1, tab2 = st.tabs(["📤 Upload & Process", "🕵️ Human Review"])

    with tab1:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Choose PDF, JPG, or PNG files", type=["pdf", "jpg", "png"], accept_multiple_files=True)
        st.warning("⚠️ RAM Note: Process 10-20 files at a time.")

        if st.button("Start Processing"):
            if not api_key:
                st.error("Please provide a LlamaCloud API Key in the sidebar.")
            elif not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                selected_prompt = PROMPT_LIBRARY[selected_mode]
                parser = LlamaParse(
                    api_key=api_key,
                    result_type="markdown",
                    parsing_instruction=selected_prompt
                )

                temp_dir = tempfile.mkdtemp()
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save to temp file
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Dynamic prompt for photos
                    current_instruction = selected_prompt
                    if uploaded_file.name.lower().endswith((".jpg", ".png", ".jpeg")):
                        current_instruction += "\nIgnore background objects like fingers/shadows."
                        parser.parsing_instruction = current_instruction

                    try:
                        # Parse
                        documents = parser.load_data(file_path)
                        result_text = "\n\n".join([doc.text for doc in documents])

                        # Quality Gate
                        if len(result_text) < 100:
                            st.session_state.review_queue.append({
                                "name": uploaded_file.name,
                                "path": file_path,
                                "type": uploaded_file.type,
                                "content": result_text
                            })
                            st.warning(f"Low quality detection for {uploaded_file.name}. Added to Human Review.")
                        else:
                            st.session_state.processed_data[uploaded_file.name] = result_text
                            st.success(f"Successfully processed {uploaded_file.name}")
                            os.remove(file_path) # Only remove if processed successfully

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("Processing complete!")
                # Note: We don't delete temp_dir here because review_queue might need files in it.
                # We should handle cleanup better in a real app, but for now we follow the prompt.

    with tab2:
        st.header("Human Review Workflow")
        st.write("Items in this queue had low-quality extraction and need manual verification.")
        
        if not st.session_state.review_queue:
            st.info("Review queue is empty.")
        else:
            st.write(f"You have {len(st.session_state.review_queue)} items to review.")
            
            # Selector for review items
            item_names = [item["name"] for item in st.session_state.review_queue]
            selected_item_name = st.selectbox("Select document to review", item_names)
            
            # Find the selected item
            item_index = item_names.index(selected_item_name)
            item = st.session_state.review_queue[item_index]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Document")
                try:
                    if item["name"].lower().endswith(".pdf"):
                        images = convert_from_path(item["path"])
                        for img in images:
                            st.image(img, use_container_width=True)
                    else:
                        st.image(item["path"], use_container_width=True)
                except Exception as e:
                    st.error(f"Could not display image: {e}")
            
            with col2:
                st.subheader("Extracted Text (Edit if needed)")
                edited_text = st.text_area("Transcription", value=item["content"], height=600)
                
                if st.button("Approve & Save"):
                    st.session_state.processed_data[item["name"]] = edited_text
                    st.session_state.review_queue.pop(item_index)
                    st.success(f"Approved and saved {item['name']}")
                    st.rerun()



if __name__ == "__main__":
    main()
