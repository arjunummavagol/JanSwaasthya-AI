# app_ui.py
import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="JanSwaasthya AI - Clinical Simplifier",
    page_icon="🩺",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/process-document"

st.title("🩺 JanSwaasthya AI")
st.caption("AI-Powered Clinical Report Simplification & Indic Language Translation")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    language_options = {
        "Kannada (ಕನ್ನಡ)": "kan",
        "Hindi (हिंदी)": "hin",
        "English (Plain English)": "en"
    }
    selected_lang_label = st.selectbox("Target Regional Language", list(language_options.keys()))
    target_lang_code = language_options[selected_lang_label]
    st.markdown("---")
    st.info("FastAPI backend should be active on `http://127.0.0.1:8000`.")

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.subheader("1. Upload Clinical Report")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"]
    )

with col_preview:
    st.subheader("Document Preview")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        try:
            st.image(image, caption="Uploaded Document", width="stretch")
        except TypeError:
            st.image(image, caption="Uploaded Document")
    else:
        st.info("Upload an image to see the preview here.")

st.markdown("---")

try:
    execute_btn = st.button("🚀 Process Document", type="primary", width="stretch")
except TypeError:
    execute_btn = st.button("🚀 Process Document", type="primary")

if execute_btn:
    if uploaded_file is None:
        st.warning("Please upload an image file first.")
    else:
        with st.spinner("Executing 4-stage pipeline..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                data = {
                    "target_language": target_lang_code
                }
                
                response = requests.post(API_URL, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Resolve data whether flat or inside pipeline_execution
                    pipeline = result.get("pipeline_execution", result)
                    
                    raw_text = pipeline.get("raw_extracted_text", result.get("original_extracted_text", ""))
                    entities = pipeline.get("extracted_clinical_entities", result.get("key_medical_entities", []))
                    simplified = pipeline.get("simplified_english", "")
                    translated = pipeline.get("final_translated_output", result.get("translated_output", ""))
                    
                    st.success("Processing Complete!")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.subheader("📄 Phase 1: Extracted Raw Clinical Text")
                        st.text_area(
                            label="Raw OCR Output",
                            value=raw_text,
                            height=140
                        )

                        st.subheader("🏷️ Phase 2: Key Clinical Entities")
                        if entities:
                            st.write(" ".join([f"`{entity}`" for entity in entities]))
                        else:
                            st.caption("No isolated clinical keywords found.")

                        st.subheader("💡 Phase 3: Simplified English")
                        st.info(simplified if simplified else "No simplification generated.")

                    with res_col2:
                        st.subheader(f"🌐 Phase 4: Regional Translation ({selected_lang_label})")
                        st.success(translated if translated else "No translation generated.")

                else:
                    st.error(f"Backend error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Verify Uvicorn is running on http://127.0.0.1:8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")