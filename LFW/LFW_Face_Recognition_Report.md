# LFW Report
# Sentiment Analysis Chatbot using NLP - Project Report

**Name:**  Bhuvan Jangid 
**Reg No:** 23BAI10372

---

## 1. Project Overview

This project develops a sentiment-aware conversational agent using Natural Language Processing (NLP). A Logistic Regression classifier trained on the Sentiment140 Twitter dataset is deployed via an interactive Streamlit web interface.

**Dataset:** Sentiment140 — 100,000 tweet sample (from 1.6M tweets)  
**Classes:** Binary — Positive (1) / Negative (0)  
**Tech Stack:** Python, Scikit-learn (TF-IDF + Logistic Regression), Streamlit, Pandas, Regex

---

## 2. Methodology

### Data Preprocessing
- Loaded 100K random sample for web app performance (`random_state=42`)
- Label conversion: Original 4→1 (positive), 0→0 (negative)
- Text cleaning via regex:
  - Remove URLs (`http...`, `www...`, `https...`)
  - Remove mentions (`@username`) and hashtags (`#tag`)
  - Convert to lowercase
- Feature extraction: `TfidfVectorizer(max_features=10000, ngram_range=(1,2))`
  - Unigrams + bigrams
  - 10,000 max features

### Model Training
- Split: 80% train, 20% test (`random_state=42`)
- Algorithm: `LogisticRegression(max_iter=500)`
- Caching: `@st.cache_resource` ensures single training per Streamlit session

### Web Interface (Streamlit)
- **Session State:** Stores conversation history
- **Chat Input:** Real-time user message capture
- **Prediction Pipeline:**
  1. Transform input via fitted TF-IDF vectorizer
  2. Predict sentiment (0/1) + confidence score
  3. Generate contextual bot response
- **UI Components:**
  - Title + developer attribution
  - Chat history display (user/assistant bubbles)
  - Real-time chat input
  - Confidence percentage in responses

---

## 3. Model Performance

| Metric | Expected Range |
|--------|----------------|
| **Training Accuracy** | ~80-85% |
| **Test Accuracy** | ~78-82% |
| **Inference Latency** | <100ms per message |

Logistic Regression with TF-IDF provides fast, interpretable baseline for sentiment classification.

---

## 4. Deployment

The Streamlit app (`sentiment_chatbot.py`) runs locally:

```bash
streamlit run sentiment_chatbot.py
# Access at http://localhost:8501
```

**Features:**
- Conversation memory (session_state)
- Dynamic bot responses with emoji
- Confidence score display
- Clean, centered UI layout

---

## 5. How to Run

```bash
# 1. Install dependencies
pip install streamlit pandas scikit-learn

# 2. Download Sentiment140 dataset
# Place 'training.1600000.processed.noemoticon.csv' in project root
# Source: https://www.kaggle.com/datasets/kazanova/sentiment140

# 3. Run the notebook to generate sentiment_chatbot.py
# 4. Launch Streamlit app
streamlit run sentiment_chatbot.py
```

---

## 6. Conclusion

The sentiment chatbot demonstrates an end-to-end NLP pipeline: data cleaning → feature engineering → model training → web deployment. Logistic Regression with TF-IDF offers a lightweight, fast solution suitable for real-time chat applications. Future improvements could include transformer-based models (BERT, DistilBERT) for higher accuracy.


