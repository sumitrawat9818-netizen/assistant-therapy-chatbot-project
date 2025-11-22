import streamlit as st
import pickle
import os
import sys
from google import genai
from clean_text import clean_text 

# --- 1. UI Setup ---
st.set_page_config(page_title="Clarity: AI Therapy", layout="centered")
st.title("🤖 Clarity: Mental Health Assistant")

# --- 2. LLM Setup ---
try:
    # The client automatically looks for "GEMINI_API_KEY" in Streamlit Secrets
    client = genai.Client()
except Exception:
    client = None

def generate_therapy_response(user_input, predicted_intent):
    if not client:
        return "Error: API Key missing. Please set GEMINI_API_KEY in Secrets."
    
    PROMPT = f"""
    You are a compassionate AI assistant. The user feels: {predicted_intent}.
    User said: "{user_input}"
    
    Provide a 3-sentence supportive response:
    1. Validate their feeling.
    2. Offer a simple coping tip.
    3. Suggest professional help if needed.
    """
    try:
        # FIXED: Using the specific stable version to avoid 404 errors
        response = client.models.generate_content(
            model='gemini-1.5-flash-001',
            contents=PROMPT
        )
        return response.text
    except Exception as e:
        return f"Connection Error: {e}"

# --- 3. Load Models with Status ---
with st.status("System Startup...", expanded=False) as status:
    try:
        st.write("Loading Intent Model...")
        with open('intent_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        st.write("Loading Vectorizer...")
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        status.update(label="System Ready!", state="complete", expanded=False)
    except FileNotFoundError:
        st.error("Critical Error: .pkl files missing! Please upload them to GitHub.")
        st.stop()

# --- 4. Chat Logic ---
user_input = st.text_area("How are you feeling right now?", height=100)

if st.button("Get Support"):
    if user_input:
        # Step A: ML Prediction
        status_text = st.empty()
        status_text.info("Step 1/2: Analyzing sentiment pattern...")
        
        cleaned_text = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_text])
        predicted_intent = model.predict(input_vec)[0]
        
        # Show result immediately
        st.sidebar.success(f"✅ Detected Intent: *{predicted_intent.upper()}*")
        
        # Step B: AI Generation
        status_text.info("Step 2/2: Generating compassionate response...")
        
        ai_response = generate_therapy_response(user_input, predicted_intent)
        
        # Clear status and show result
        status_text.empty()
        st.subheader("Clarity's Response:")
        st.write(ai_response)
        
    else:
        st.warning("Please type something first.")
