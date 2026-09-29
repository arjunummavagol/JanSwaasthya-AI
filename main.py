# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from modules.preprocessing import Preprocessor
from modules.simplifier import MedicalSimplifier
from modules.translator import IndicTranslator

app = FastAPI(
    title="JanSwaasthya AI API",
    description="Clinical Notes Simplification and Translation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = Preprocessor(tfidf_threshold=0.05)
simplifier = MedicalSimplifier()
translator = IndicTranslator()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "project": "JanSwaasthya AI"
    }

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    target_language: str = Form("kan")
):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Phase 1: Ingestion & Preprocessing[cite: 1]
        raw_text = preprocessor.extract_text(temp_path)
        clean_text = preprocessor.clean_text(raw_text)
        tokens = preprocessor.tokenize_and_remove_stopwords(clean_text)

        # Phase 2: Feature Selection (Filter Method)[cite: 1]
        key_entities = preprocessor.select_features(tokens)

        # Phase 3: Simplification[cite: 1]
        simplified_english = simplifier.simplify(clean_text, key_entities)

        # Phase 4: Translation[cite: 1]
        translated_output = translator.translate(simplified_english, target_lang=target_language)

        payload = {
            "status": "success",
            "target_language": target_language,
            "raw_extracted_text": raw_text.strip(),
            "cleaned_text": clean_text,
            "extracted_clinical_entities": key_entities,
            "simplified_english": simplified_english,
            "final_translated_output": translated_output
        }
        
        # Return both flat and nested for backwards compatibility
        payload["pipeline_execution"] = payload.copy()
        return JSONResponse(status_code=200, content=payload)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)}
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)