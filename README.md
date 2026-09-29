# JanSwaasthya AI 🩺

JanSwaasthya AI is an automated clinical jargon simplification and regional language translation pipeline. It ingests digital medical records and prescriptions, translates dense medical jargon into accessible plain English, and localizes the output into regional Indian languages (Kannada, Hindi) for improved post-operative patient care.

## Project Objective
To bridge the health literacy gap caused by complex medical English. By providing a 4-stage sequential NLP architecture, the system empowers non-English-speaking patients to understand their discharge summaries, medication dosages, and care instructions, thereby reducing medical non-compliance and preventable hospital readmissions.

## System Architecture
1. **Data Ingestion & Preprocessing:** Extracts text from images via Tesseract OCR, cleans noise with Regex, and tokenizes using spaCy.
2. **Feature Selection:** Utilizes a TF-IDF filter method to isolate high-value clinical entities (e.g., symptoms, medications).
3. **Generative Simplification:** Uses a fine-tuned Large Language Model (Llama-3 + LoRA) or a deterministic lexical engine to rewrite complex jargon into plain English.
4. **Neural Machine Translation (NMT):** Translates the simplified English into target regional languages utilizing IndicTrans2 and domain-specific lexicons.

Prerequisites
-------------

*   **OS:** Windows 10/11 or Ubuntu
    
*   **Python:** 3.9 - 3.12
    
*   **Tesseract OCR:** Required for image text extraction.
    
*   **Hugging Face Token:** (HF\_TOKEN) Required if accessing gated models (Llama-3, IndicTrans2). The system automatically falls back to deterministic lexical mapping if no token or GPU is available locally.
    

Local Setup Instructions
------------------------

**1\. Install Tesseract OCR (Windows)**

*   Download and install the [UB-Mannheim Tesseract Windows installer](https://www.google.com/search?q=https://github.com/UB-Mannheim/tesseract/wiki).
    
*   Ensure it is installed to the default directory: C:\\Program Files\\Tesseract-OCR.
    

**2\. Set Up the Python Environment**Open your terminal (PowerShell or Git Bash) and navigate to the project folder:

Bash
`   python -m venv venv   `

_Activate the environment:_

*   Windows (PowerShell): .\\venv\\Scripts\\Activate.ps1
    
*   Git Bash/Linux: source venv/Scripts/activate
    

**3\. Install Dependencies**

Bash
`   pip install -r requirements.txt  python -m spacy download en_core_web_sm   `

Running the Application
-----------------------

The project uses a decoupled architecture. You must run the backend API and the frontend UI simultaneously in separate terminals.

**Terminal 1: Start the FastAPI Backend**Ensure your virtual environment is active, then run:

Bash
`   uvicorn main:app --host 127.0.0.1 --port 8000   `

_The backend API will be available at http://127.0.0.1:8000._

**Terminal 2: Start the Streamlit Frontend**Open a new terminal, activate the virtual environment, and run:

Bash
`   streamlit run app_ui.py   `

_The interactive dashboard will automatically open in your default browser at http://localhost:8501._

``*(Note: Before you run `git add .` and `git commit`, make sure you have created a `.gitignore` file containing `venv/` so Git ignores your virtual environment folder!)*``

## Project Structure
```text
janswaasthya_ai/
├── main.py                  # FastAPI backend orchestrator and REST endpoints
├── app_ui.py                # Streamlit frontend dashboard
├── requirements.txt         # Python package dependencies
├── generate_docs.py         # Script to auto-generate the project report and PPT
├── README.md                # Project documentation and setup guide
├── .gitignore               # Ignored files (e.g., venv, temporary files)
├── modules/                 # Core AI and processing modules
│   ├── __init__.py          
│   ├── preprocessing.py     # Tesseract OCR, spaCy tokenization, TF-IDF filter
│   ├── simplifier.py        # Llama-3/LoRA generative simplification engine
│   └── translator.py        # IndicTrans2 regional neural machine translation
└── venv/                    # Python virtual environment (Do not commit to Git)


