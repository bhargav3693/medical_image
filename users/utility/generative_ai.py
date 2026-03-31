import google.generativeai as genai

# Initialize with my Google Gemini Key
genai.configure(api_key="AIzaSyDIhh7z94PkkV_s_VXftiW7qPHNFahVDIM")

def get_clinical_advice(disease_name, dataset_type):
    try:
        # Using the fast flash model
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"You are a professional medical AI assistant. A patient's {dataset_type} scan indicates: {disease_name}. Provide exactly 2 short, concise sentences: one stating the clinical suggestion, and one stating the immediate precaution. Do not use markdown like bold text or asterisks."
        
        response = model.generate_content(prompt)
        
        if response.text:
            # Clean up any potential markdown asterisks that Gemini might still output
            clean_text = response.text.replace('*', '').strip()
            return clean_text
        else:
            return "Please consult a doctor for further clinical suggestions and immediate medical precautions."
            
    except Exception as e:
        return f"Gemini API Error: {str(e)}"
