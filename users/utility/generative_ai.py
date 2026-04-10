def get_clinical_advice(disease_name, dataset_type):
    from google import genai
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        # Initialize client inside the function to prevent Gunicorn worker boot delays/OOM
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Clinical advice is currently unavailable. Please configure your GEMINI_API_KEY in the .env file."
            
        client = genai.Client(api_key=api_key)
        
        prompt = f"You are a professional medical AI assistant. A patient's {dataset_type} scan indicates: {disease_name}. Provide exactly 2 short, concise sentences: one stating the clinical suggestion, and one stating the immediate precaution. Do not use markdown like bold text or asterisks."
        
        # Using the standard 2.5 flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        if response.text:
            # Clean up any potential markdown asterisks that Gemini might still output
            clean_text = response.text.replace('*', '').strip()
            return clean_text
        else:
            return "Please consult a doctor for further clinical suggestions and immediate medical precautions."
            
    except Exception as e:
        return f"Gemini API Error: {str(e)}"
