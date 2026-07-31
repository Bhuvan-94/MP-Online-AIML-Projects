import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import shap
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
import requests
import subprocess
import sys

matplotlib.use('Agg')

# Configurations
MODEL_DIR = "models"
DATA_PATH = "data/raw/adult.csv"
PIPELINE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# ==============================================================================
# 1. Data Loader & Preprocessing
# ==============================================================================
def load_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    
    print("Downloading Adult dataset...")
    data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
    df = data.frame
    df['class'] = df['class'].apply(lambda x: 1 if x == '>50K' else 0)
    df = df.replace('?', np.nan)
    df.to_csv(DATA_PATH, index=False)
    return df

def build_preprocessor():
    num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# ==============================================================================
# 2. Training
# ==============================================================================
def run_train():
    df = load_data()
    X = df.drop(columns=['class'])
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    
    print("Training model...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                              random_state=42, scale_pos_weight=3.0, eval_metric='auc')
    model.fit(X_train_processed, y_train)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PIPELINE_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Training complete. Artifacts saved.")

# ==============================================================================
# 3. Evaluation & Explanations
# ==============================================================================
def run_evaluate():
    df = load_data()
    X = df.drop(columns=['class'])
    y = df['class']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with open(PIPELINE_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Metrics
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    
    # Plots
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.close()
    
    explainer = shap.TreeExplainer(model)
    sample_indices = shap.sample(X_test_processed, 100)
    shap_values = explainer.shap_values(sample_indices)
    
    cat_out = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    features = list(preprocessor.transformers_[0][2]) + list(cat_out)
    
    plt.figure()
    shap.summary_plot(shap_values, sample_indices, feature_names=features, show=False)
    plt.savefig(os.path.join(MODEL_DIR, 'shap_summary.png'), bbox_inches='tight')
    plt.close()
    print("Evaluation and explanations saved.")

# ==============================================================================
# 4. FastAPI Setup
# ==============================================================================
app = FastAPI(title="Adult Census Income Classifier API")

class PredictionRequest(BaseModel):
    age: int = 30
    workclass: str = "Private"
    fnlwgt: int = 100000
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Never-married"
    occupation: str = "Tech-support"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = 0
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = "United-States"

try:
    with open(PIPELINE_PATH, 'rb') as f: api_preprocessor = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f: api_model = pickle.load(f)
except:
    api_preprocessor, api_model = None, None

@app.post("/predict")
def predict(request: PredictionRequest):
    if not api_model: raise HTTPException(status_code=503, detail="Model not loaded.")
    input_data = {
        'age': [request.age], 'workclass': [request.workclass], 'fnlwgt': [request.fnlwgt],
        'education': [request.education], 'education-num': [request.education_num],
        'marital-status': [request.marital_status], 'occupation': [request.occupation],
        'relationship': [request.relationship], 'race': [request.race], 'sex': [request.sex],
        'capital-gain': [request.capital_gain], 'capital-loss': [request.capital_loss],
        'hours-per-week': [request.hours_per_week], 'native-country': [request.native_country],
    }
    df = pd.DataFrame(input_data)
    X_processed = api_preprocessor.transform(df)
    proba = api_model.predict_proba(X_processed)[0]
    pred = api_model.predict(X_processed)[0]
    return {"prediction": ">50K" if pred == 1 else "<=50K", "propensity": float(proba[1])}

# ==============================================================================
# 5. Main Execution / Streamlit Logic
# ==============================================================================
if __name__ == '__main__':
    # Checking if executed via Streamlit
    if 'streamlit' in sys.argv[0] or (len(sys.argv) > 1 and sys.argv[1] == '--mode' and sys.argv[2] == 'ui'):
        # Streamlit execution payload
        st.title("Adult Census Income Prediction")
        
        with st.form("pred_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", 17, 90, 30)
                workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov'])
                fnlwgt = st.number_input("Final Weight", value=100000)
                education = st.selectbox("Education", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Masters'])
                education_num = st.number_input("Education Num", min_value=1, max_value=16, value=9)
                marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married'])
                occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial'])
            with col2:
                relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
                race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
                sex = st.selectbox("Sex", ['Female', 'Male'])
                capital_gain = st.number_input("Capital Gain", value=0)
                capital_loss = st.number_input("Capital Loss", value=0)
                hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
                native_country = st.selectbox("Native Country", ['United-States', 'India', 'Japan', 'Mexico', 'Canada'])
            
            submitted = st.form_submit_button("Predict Income Tier")
            
        if submitted:
            payload = {
                "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
                "education": education, "education_num": education_num,
                "marital_status": marital_status, "occupation": occupation,
                "relationship": relationship, "race": race, "sex": sex,
                "capital_gain": capital_gain, "capital_loss": capital_loss,
                "hours_per_week": hours_per_week, "native_country": native_country
            }
            try:
                res = requests.post("http://localhost:8000/predict", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"**Prediction:** {data['prediction']}")
                    st.info(f"**Propensity Score (>50K):** {data['propensity']:.4f}")
                else:
                    st.error("API error!")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Please ensure you are running the API with `--mode api` on port 8000.")

    else:
        # CLI execution payload
        parser = argparse.ArgumentParser(description="Adult Census Income Monolithic App")
        parser.add_argument('--mode', choices=['train', 'evaluate', 'api', 'ui'], required=True, 
                            help="Mode to run the program in.")
        args, _ = parser.parse_known_args()
        
        if args.mode == 'train':
            run_train()
        elif args.mode == 'evaluate':
            run_evaluate()
        elif args.mode == 'api':
            print("Starting FastAPI on port 8000...")
            uvicorn.run(app, host="0.0.0.0", port=8000)
        elif args.mode == 'ui':
            print("Starting Streamlit Dashboard...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--", "--mode", "ui"])
