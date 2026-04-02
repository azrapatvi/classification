import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

model = pickle.load(open('models/model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def clean_message(text):
    cleaned = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    pos_tags = nltk.pos_tag(cleaned)
    cleaned = [lemmatizer.lemmatize(w, pos=get_wordnet_pos(p)) for w, p in pos_tags if w not in stop_words]
    return ' '.join(cleaned)

def predict(text):
    cleaned = clean_message(text)
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = 'Harassment' if pred == 1 else 'Normal'
    confidence = proba[1] if pred == 1 else proba[0]
    return label, round(confidence * 100, 2)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛡️ Text Harassment Classifier")
st.write("Detects whether a social media comment is **harassment** or **normal**.")

text = st.text_area("Enter a message:", height=150)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        label, confidence = predict(text)
        if label == "Harassment":
            st.error(f"🚨 **{label}** — Confidence: {confidence}%")
        else:
            st.success(f"✅ **{label}** — Confidence: {confidence}%")