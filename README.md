# 🛡️ Text Harassment Classifier

A machine learning web app that detects whether a social media comment is **harassment** or **normal**, built with Logistic Regression, TF-IDF, and Streamlit.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web app
├── classification.ipynb    # Model training notebook
├── model.pkl               # Trained Logistic Regression model
├── tfidf.pkl               # Fitted TF-IDF vectorizer
├── data/
│   └── dataset.csv         # Raw dataset (from Kaggle)
└── requirements.txt
```

---

## 📊 Dataset

- **Source:** Kaggle — social media comments from Facebook, TikTok, YouTube, and Instagram
- **Raw rows:** 8,452 | **Final rows after cleaning:** 6,075
- **Classes:** `harassment` (4,834) · `normal` (3,616)
- **Cleaning steps:**
  - Consolidated 19 label variants (typos like `Not-Bulllying`) into 2 classes
  - Dropped `Types` column (43% missing values)
  - Removed 2 null-label rows and 2,375 duplicate rows
  - Lowercased all column names

---

## ⚙️ ML Pipeline

| Step | Detail |
|------|--------|
| Regex Cleaning | Remove all non-alphabetic characters |
| Lowercasing | Normalize text |
| Tokenization | Split into words |
| Stopword Removal | NLTK English stopwords |
| POS Lemmatization | WordNetLemmatizer with POS tags |
| Vectorization | TF-IDF, `max_features=2500`, `ngram_range=(1,2)` |
| Train/Test Split | 80/20 → 4,860 train / 1,215 test |

---

## 🤖 Model

**Logistic Regression** selected after comparing 4 classifiers:

| Model | Train Acc | Test Acc |
|-------|-----------|----------|
| **Logistic Regression ✓** | ~87% | **86.67%** |
| Random Forest | 97.6% | 84.6% |
| SVM | — | ~85% |
| Naive Bayes | — | ~82% |

**Best params** via GridSearchCV (5-fold, F1 scoring):  
`C=1, penalty='l2', solver='liblinear', max_iter=500`

**Results:**

| Metric | Score |
|--------|-------|
| Accuracy | 86.67% |
| F1-Score | 88% |
| Precision | ~87% |
| Recall | ~89% |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/text-harassment-classifier.git
cd text-harassment-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Export model from notebook
Run this at the end of `classification.ipynb`:
```python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit>=1.32.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8.0
```

---

## 🖥️ App Preview

Enter any text → click **Predict** → get result:

- 🚨 **Harassment** — Confidence: 91.3%
- ✅ **Normal** — Confidence: 87.6%

---

## 🔮 Future Improvements

- Apply SMOTE / `class_weight='balanced'` for class imbalance
- Replace TF-IDF with BERT / multilingual-BERT embeddings
- Add multilingual tokenizer for Bengali and mixed-script text
- Ensemble with LinearSVC
- Batch CSV prediction support
- Threshold tuning for precision/recall tradeoff
