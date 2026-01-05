# 🇮🇳 Indian Document Digitizer

A powerful Streamlit application designed to digitize complex Indian documents (Legal, Astrological, Property, Revenue, Banking) using AI-powered parsing with LlamaParse.

## ✨ Features

- **🎯 Specialized Mode Switching**: Custom parsing instructions for different document types:
  - **⚖️ Legal**: Forensic transcription of court orders, stamps, and marginalia.
  - **🔮 Astrology**: Expert handling of South/North Indian charts and planet positions.
  - **🏠 Property Deeds**: Extraction of boundaries, schedules, and handling of thumb impressions.
  - **🏛️ Revenue Records**: Preservation of grid structures and handwritten remarks.
  - **💰 Banking**: Optimized for dot-matrix passbook prints.
- **🧠 Intelligent Processing**: Uses LlamaParse for high-quality markdown extraction with dynamic photo logic to ignore background artifacts.
- **🕵️ Human Review Workflow**: Built-in quality gate that flags low-quality extractions for manual verification and editing.
- **📦 Large File Support**: Configured to handle scans up to 2GB.
- **📥 Easy Export**: Download all processed documents as a single ZIP file.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- **Poppler** (Required for PDF preview in Human Review):
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/amruth9459/Indian-Docs-Digitizer.git
   cd Indian-Docs-Digitizer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
```bash
streamlit run app.py
```

## ☁️ Deployment

This app is optimized for **Streamlit Community Cloud**. 
1. Push your code to GitHub.
2. Connect your repository to [Streamlit Cloud](https://share.streamlit.io).
3. The `packages.txt` file will automatically handle system-level dependencies (Poppler, Tesseract) on the cloud server.

## 🛠️ Built With
- [Streamlit](https://streamlit.io/) - The web framework.
- [LlamaParse](https://github.com/run-llama/llama_parse) - AI-powered document parsing.
- [pdf2image](https://github.com/Belval/pdf2image) - PDF to image conversion.

## 📄 License
This project is licensed under the MIT License.
