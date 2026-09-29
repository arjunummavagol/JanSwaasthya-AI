# modules/simplifier.py
import re
import os

class MedicalSimplifier:
    def __init__(self, base_model_name: str = "meta-llama/Meta-Llama-3-8B", lora_weights_path: str = "./models/medical_lora"):
        self.is_loaded = False
        hf_token = os.environ.get("HF_TOKEN")

        # Only attempt remote download if a token is explicitly configured
        if hf_token and os.path.exists(lora_weights_path):
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from peft import PeftModel

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    token=hf_token,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self.model = PeftModel.from_pretrained(base_model, lora_weights_path)
                self.is_loaded = True
            except Exception:
                self.is_loaded = False

        self.jargon_map = {
            "myocardial infarction": "heart attack",
            "hypertension": "high blood pressure",
            "hyperlipidemia": "high cholesterol",
            "cerebrovascular accident": "stroke",
            "tachycardia": "fast heart rate",
            "bradycardia": "slow heart rate",
            "dyspnea": "shortness of breath",
            "edema": "swelling",
            "erythema": "skin redness",
            "prn": "as needed",
            "po": "by mouth (orally)",
            "qd": "once every day",
            "bid": "two times a day",
            "tid": "three times a day",
            "qid": "four times a day",
            "qhs": "at bedtime every night",
            "analgesic": "pain reliever",
            "antipyretic": "fever reducer",
            "nephropathy": "kidney damage",
            "benign": "non-cancerous",
            "malignant": "cancerous"
        }

    def simplify(self, context_text: str, key_entities: list[str]) -> str:
        if self.is_loaded:
            import torch
            prompt = (
                f"Below is a complex clinical summary. Rewrite it in plain English for a patient:\n"
                f"Clinical Text: {context_text}\n"
                f"Key Entities: {', '.join(key_entities)}\n"
                f"Simplified Patient Instructions:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=False
                )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.split("Simplified Patient Instructions:")[-1].strip()

        simplified = context_text
        for jargon, plain in self.jargon_map.items():
            pattern = re.compile(re.escape(jargon), re.IGNORECASE)
            simplified = pattern.sub(f"{plain} ({jargon})", simplified)

        return (
            f"Patient Medical Summary:\n"
            f"{simplified}\n\n"
            f"Instructions: Please follow the prescribed doses carefully. "
            f"If discomfort or unusual symptoms persist, visit the healthcare center immediately."
        )