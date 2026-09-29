# modules/translator.py
import os
import re

class IndicTranslator:
    def __init__(self, model_name: str = "ai4bharat/indictrans2-en-indic-1B"):
        self.is_loaded = False
        hf_token = os.environ.get("HF_TOKEN")

        # Only attempt remote download if Hugging Face token is provided
        if hf_token:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    trust_remote_code=True, 
                    token=hf_token
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=hf_token,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.is_loaded = True
            except Exception:
                self.is_loaded = False

        # Direct domain translation mapping
        self.regional_lexicon = {
            "kan": {
                "heart attack": "ಹೃದಯಾಘಾತ",
                "high blood pressure": "ಅಧಿಕ ರಕ್ತದೊತ್ತಡ",
                "high cholesterol": "ಹೆಚ್ಚಿನ ಕೊಲೆಸ್ಟ್ರಾಲ್",
                "stroke": "ಪಾರ್ಶ್ವವಾಯು",
                "shortness of breath": "ಉಸಿರಾಟದ ತೊಂದರೆ",
                "swelling": "ಊತ",
                "skin redness": "ಚರ್ಮದ ಕೆಂಪಾಗುವಿಕೆ",
                "pain reliever": "ನೋವು ನಿವಾರಕ ಔಷಧಿ",
                "fever reducer": "ಜ್ವರ ನಿಯಂತ್ರಣ ಔಷಧಿ",
                "once every day": "ದಿನಕ್ಕೆ ಒಂದು ಬಾರಿ",
                "two times a day": "ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ",
                "three times a day": "ದಿನಕ್ಕೆ ಮೂರು ಬಾರಿ",
                "at bedtime every night": "ಪ್ರತಿದಿನ ರಾತ್ರಿ ಮಲಗುವ ಮುನ್ನ",
                "header": "ರೋಗಿಯ ವೈದ್ಯಕೀಯ ಸಾರಾಂಶ ಮತ್ತು ಸೂಚನೆಗಳು (Kannada):",
                "disclaimer": "ಪ್ರಮುಖ ಸೂಚನೆ: ಸೂಚಿಸಲಾದ ಔಷಧಿಗಳ ಪ್ರಮಾಣವನ್ನು ಸರಿಯಾಗಿ ಪಾಲಿಸಿ. ಯಾವುದೇ ಗಂಭೀರ ತೊಂದರೆಗಳಿದ್ದಲ್ಲಿ ತಕ್ಷಣವೇ ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ."
            },
            "hin": {
                "heart attack": "दिल का दौरा",
                "high blood pressure": "उच्च रक्तचाप",
                "high cholesterol": "उच्च कोलेस्ट्रॉल",
                "stroke": "स्ट्रोक (दौरा)",
                "shortness of breath": "सांस लेने में कठिनाई",
                "swelling": "सूजन",
                "skin redness": "त्वचा का लाल होना",
                "pain reliever": "दर्द निवारक दवा",
                "fever reducer": "बुखार की दवा",
                "once every day": "दिन में एक बार",
                "two times a day": "दिन में दो बार",
                "three times a day": "दिन में तीन बार",
                "at bedtime every night": "हर रात सोने से पहले",
                "header": "रोगी का मेडिकल सारांश और निर्देश (Hindi):",
                "disclaimer": "महत्वपूर्ण सूचना: दवाओं की खुराक का सही समय पर पालन करें। किसी भी समस्या के मामले में तुरंत डॉक्टर से संपर्क करें।"
            }
        }

    def translate(self, simplified_text: str, target_lang: str = "kan") -> str:
        """Translates simplified English into regional languages like Kannada or Hindi."""
        target = target_lang.lower().strip()
        if target in ["en", "english"]:
            return simplified_text

        if self.is_loaded:
            import torch
            inputs = self.tokenizer(simplified_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=512)
            translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translated

        lang_dict = self.regional_lexicon.get(target, self.regional_lexicon["kan"])
        translated_body = simplified_text
        for eng_term, regional_term in lang_dict.items():
            if eng_term not in ["header", "disclaimer"]:
                pattern = re.compile(re.escape(eng_term), re.IGNORECASE)
                translated_body = pattern.sub(f"{regional_term} ({eng_term})", translated_body)

        return (
            f"{lang_dict['header']}\n\n"
            f"{translated_body}\n\n"
            f"{lang_dict['disclaimer']}"
        )