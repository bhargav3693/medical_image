def get_clinical_advice(disease_name, dataset_type):
    from google import genai
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file (works locally)
    load_dotenv()

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Please consult a doctor for clinical suggestions and immediate medical precautions."

        client = genai.Client(api_key=api_key)

        prompt = (
            f"You are a professional medical AI assistant. "
            f"A patient's {dataset_type} scan indicates: {disease_name}. "
            f"Provide exactly 2 short, concise sentences: one stating the clinical suggestion, "
            f"and one stating the immediate precaution. Do not use markdown, bold, or asterisks."
        )

        # gemini-1.5-flash: stable, fast, widely available
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )

        if response.text:
            clean_text = response.text.replace('*', '').replace('#', '').strip()
            return clean_text
        else:
            return "Please consult a doctor for further clinical suggestions and immediate medical precautions."

    except Exception as e:
        # NEVER crash the prediction — just return fallback text
        print(f"[Gemini Warning] {str(e)}")
        return "Please consult a qualified doctor for clinical suggestions and immediate medical precautions."
