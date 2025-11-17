import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import os
import sys

# Setting up NLTK resource paths for cloud deployment
# This is a robust attempt to tell the system where to look for data files.
# CRITICAL FIX 1: Explicitly add the local user path where downloads often land
nltk.data.path.append(os.path.join(os.path.expanduser("~"), 'nltk_data'))

# IMPORTANT: Download the required NLTK data when the script starts
try:
    # CRITICAL FIX 2: We run the downloads silently to avoid the Streamlit environment crash.
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    # If downloading fails, the script continues and looks for the data in the path added above.
    pass

# Initialize the global objects once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Applies the exact same cleaning steps used during model training: 
    1. Removes special characters/numbers (noise reduction).
    2. Converts to lowercase (standardization).
    3. Tokenizes, Lemmatizes (word root form), and removes stopwords.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove non-alphabetic chars
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A) 
    text = text.lower()
    text = text.strip()
    
    # 2. Tokenize and Lemmatize
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 3. Remove Stopwords
    filtered_tokens = [word for word in lemmatized_tokens if word not in stop_words]
    
    return " ".join(filtered_tokens)

# The file ENDS here. No test blocks or extra lines below this point.
