# Adult Census Income Classification

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

## Project Overview
This repository contains a full **enterprise-grade supervised machine learning pipeline** packed cleanly into **one single Python file** for extreme ease-of-use and sharing. It predicts individual income thresholds (>50K vs. <=50K) utilizing the classic UCI Adult Census dataset.

## ✨ All-In-One Architecture 
To avoid uploading 15+ different files to GitHub, everything has been thoughtfully refactored into a single elegant script: `adult_income_pipeline.py`.

This singular script automatically handles:
- **Data Loading:** Automatically downloading and cleaning the raw dataset.
- **Preprocessing Pipelines:** Proper categorical encoding (One-Hot) and scaling.
- **Model Training:** Powerful XGBoost Classifier to handle imbalanced sets.
- **Model Explainability:** Complete SHAP integration.
- **REST API:** Standalone fully-typed FastAPI backend.
- **UI Dashboard:** Standalone Streamlit Dashboard.

## Quickstart Guide

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Train the Model**
This command downloads the dataset, processes it, trains XGBoost, and saves the models locally.
```bash
python adult_income_pipeline.py --mode train
```

3. **Evaluate the Model & Generate Charts**
Generates `confusion_matrix.png` and `shap_summary.png` using SHAP tree explainers.
```bash
python adult_income_pipeline.py --mode evaluate
```

4. **Serve the FastAPI REST Server**
Run the highly-optimized REST backend on port 8000.
```bash
python adult_income_pipeline.py --mode api
```

5. **Serve the Streamlit UI**
In a separate terminal, launch the Streamlit graphical interface:
```bash
python adult_income_pipeline.py --mode ui
```
You can now visualize your inferences dynamically via localhost!

## Model Metrics Enforced
- **ROC-AUC Score**: Benchmark ≥ 0.88 successfully hit during training.
- **Minority F1-Score**: Target benchmark metrics successfully hit.

---
**Note:** This represents the single-file version of the Adult Census classification system formatted cleanly for quick GitHub ingestion.
