import streamlit as st
import pickle
import numpy as np
import os
import sys

# Add the current directory to path to find clean_text.py
# (This is often needed for Streamlit deployments)
sys.path.append(os.path.dirname(__file__)) 
from clean_text import clean_text # Assuming you defined the full clean_text function in clean_text.py

# --- 1. LLM Setup (The Therapy Brain) ---
# NOTE: The API key must be set as a Streamlit Secret, not directly in the code!
try:
    from google import genai
    client = genai.Client()
except Exception as e:
    st.error(f"Failed to initialize Google GenAI client: {e}")
    client = None

def generate_therapy_response(user_input: str, predicted_intent: str) -> str:
    """Uses the Gemini model to generate an empathetic, therapeutic response."""
    if not client:
        return "Therapy Assistant is offline due to API key error. Please check configuration."
    
    PROMPT = f"""
    You are a compassionate, non-judgmental mental health assistant named 'Clarity'. 
    Your goal is to offer empathetic support, validation, and a gentle redirection.
    
    The user's core psychological intent has been classified as: {predicted_intent}.
    The user's statement was: "{user_input}".
    
    Based on the intent, provide a brief (2-3 sentence) response that:
    1. Validates their feeling (e.g., "It sounds like you are feeling...")
    2. Offers one simple, actionable coping strategy (e.g., a deep breath, small walk).
    3. Gently suggests that speaking with a human professional (a therapist or counselor) is the best next step.
    
    Do not use emojis. Maintain a calm, caring, and professional tone.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=PROMPT,
            config={"temperature": 0.5} 
        )
        return response.text
    except Exception as e:
        return f"I'm sorry, I'm currently unable to generate a response. Your feelings matter. Please consider reaching out to a support line. (API Error: {e})"

# --- 2. Load the ML Assets ---
try:
    with open('intent_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("ML model files (*.pkl) not found. Cannot run prediction. Ensure they are in your deployment folder.")
    st.stop()


# --- 3. Streamlit UI Setup ---
st.set_page_config(page_title="Clarity: AI Assistant Therapy Demo", layout="centered")
st.title("🤖 Clarity: AI Assistant Therapy Assistant")
st.markdown("Your companion powered by **Intent Recognition** and **Generative AI**.")


# --- 4. Chat Logic ---
user_input = st.text_area("How are you feeling today? (Tell me what's on your mind)", height=100)

if st.button('Get Support') and user_input:
    # A. Predict Intent
    st.info("Analyzing your emotional intent...")
    
    # Clean the text using the imported function
    user_cleaned = clean_text(user_input) 
    
    # Vectorize and Predict
    user_vec = vectorizer.transform([user_cleaned])
    predicted_intent = model.predict(user_vec)[0]
    
    st.sidebar.markdown(f"**ML Classified Intent:** `{predicted_intent.upper()}`")
    
    # B. Generate Therapeutic Response
    with st.spinner("Clarity is synthesizing a compassionate response..."):
        therapy_response = generate_therapy_response(user_input, predicted_intent)
    
    # C. Display Results
    st.subheader("Clarity's Response:")
    st.markdown(therapy_response)
    st.success("Please continue to talk about your feelings.")