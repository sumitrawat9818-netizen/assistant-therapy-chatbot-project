import pandas as pd
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import re

# --- 1. Setup and Preprocessing Initialization ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the file path (must match the name inside the extracted folder)
DATA_FILE_PATH = 'extracted_data/intents.json'
try:
    with open(DATA_FILE_PATH) as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"ERROR: Cannot find the data file at {DATA_FILE_PATH}. Check your extraction folder content.")
    raise

# Flatten the JSON structure
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

df = pd.DataFrame({'text': patterns, 'intent': tags})

# --- 3. NLP Cleaning Function ---
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A) # Remove non-alphabetic chars
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    filtered_tokens = [word for word in lemmatized_tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# --- 4. Feature Engineering (TF-IDF) and Data Split ---
X = df['cleaned_text']
y = df['intent']

# Split data (80% Train, 20% Test) without stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit TF-IDF Vectorizer (This converts words to numbers)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 5. Model Training (Logistic Regression - The Intent Classifier) ---
model = LogisticRegression(max_iter=1000)
print("\nTraining Logistic Regression Model...")
model.fit(X_train_vec, y_train)
print("Training Complete.")

# --- 6. Evaluation (Your Report Data!) ---
y_pred = model.predict(X_test_vec)

print("\n--- Model Performance Report (Use this for your B.Tech Report) ---")
print(f"Model Accuracy on Test Set: {model.score(X_test_vec, y_test):.4f}")
print(classification_report(y_test, y_pred, zero_division=0))
#

# --- 7. Save Final Assets (for Deployment) ---
# These are the files you download and use in your Streamlit app
with open('intent_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("\n✅ SUCCESS: Model and Vectorizer saved as intent_model.pkl and tfidf_vectorizer.pkl")
print("These two files are your entire Machine Learning project!")