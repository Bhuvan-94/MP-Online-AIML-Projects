# ML-Projects-Portfolio

A comprehensive collection of 9 machine learning, deep learning, reinforcement learning, and MLOps projects spanning computer vision, NLP, and recommender systems.

---

## 📂 Repository Architecture

```text
.
├── adult census income/              # Classification & Interpretability Pipeline
├── Cancer Classification/            # Medical Data Classification & Evaluation
├── cart pole RL agent/               # Deep Q-Network & PPO Reinforcement Learning
├── CIFAR 10/                         # CNN Image Classification Benchmark
├── End to End render deployement/    # Containerized FastAPI Web Service Deployment
├── LFW/                              # Face Recognition & Embeddings
├── Lunar Lander RL agent/            # Lunar Landing Simulation with RL
├── Movie recommendation system/      # Collaborative & Content Hybrid Recommender
├── RAG Chatbot/                      # Conversational AI with Vector Store Retrieval
└── README.md                         # This file
```

---

## 1. Adult Census Income Classification

**Predict income >50K vs ≤50K using UCI Adult Census dataset**

### Features
- Single-file monolithic pipeline (`adult_income_pipeline.py`)
- XGBoost with class imbalance handling
- SHAP explainability
- FastAPI REST API + Streamlit UI

### Quickstart
```bash
cd "adult census income/adult census income"
pip install -r requirements.txt
python adult_income_pipeline.py --mode train
python adult_income_pipeline.py --mode evaluate
python adult_income_pipeline.py --mode api    # Terminal 1
python adult_income_pipeline.py --mode ui     # Terminal 2
```

---

## 2. Cancer Classification (Brain Tumor MRI)

**Transfer Learning with MobileNetV2 for 4-class brain tumor classification**

### Features
- MobileNetV2 (ImageNet weights, frozen backbone)
- Custom classification head (~362K trainable params)
- Data augmentation, early stopping, LR scheduling
- Target: 90%+ test accuracy

### Quickstart
```bash
cd "Cancer Classification"
pip install tensorflow matplotlib numpy scipy scikit-learn seaborn opendatasets
# Run notebook: Cancer_Classification_AVNISH_AGRAWAL_23BAI10628.ipynb
```

---

## 3. Cart Pole RL Agent

**Solve CartPole-v1 with DQN (custom PyTorch) and PPO (Stable-Baselines3)**

### Features
- Custom DQN with Replay Buffer & Target Network
- Stable-Baselines3 PPO baseline
- FastAPI inference API + Streamlit dashboard
- Single-file pipeline (`cartpole_RL.py`)

### Quickstart
```bash
cd "cart pole RL agent"
pip install -r requirements.txt
python cartpole_RL.py --mode train_dqn
python cartpole_RL.py --mode train_ppo
python cartpole_RL.py --mode evaluate
python cartpole_RL.py --mode api    # Terminal 1
python cartpole_RL.py --mode ui     # Terminal 2
```

---

## 4. CIFAR-10 Image Classification

**CNN for 10-class image classification on CIFAR-10 (32x32 RGB)**

### Features
- Custom CNN architecture (~1.2M params)
- Data augmentation, batch normalization, dropout
- Target: 85%+ test accuracy (achieves ~87%)

### Quickstart
```bash
cd "CIFAR 10"
pip install tensorflow matplotlib numpy pandas scikit-learn seaborn
# Run notebook: CIFAR10_Classification_AVNISH_AGRAWAL_23BAI10628.ipynb
```

---

## 5. End-to-End Render Deployment

**Production-grade ML deployment template for Render Cloud**

### Features
- Docker containerization
- Render Blueprint (`render.yaml`) for IaC
- FastAPI with health checks (`/health`)
- Pytest test suite
- Zero-downtime deployment ready

### Quickstart
```bash
cd "End to End render deployement"
pip install -r requirements.txt
pytest test_main.py
python main_api.py --mode serve
# Docker: docker build -t render-ml-api . && docker run -p 8000:8000 render-ml-api
# Deploy: Push to GitHub → Render Blueprint
```

---

## 6. LFW Face Recognition (Sentiment Chatbot)

**Face verification and identification leveraging facial embeddings and feature extraction**

### Features
- Data pipeline for Labeled Faces in the Wild dataset
- Facial Embeddings Extraction
- Model training and verification
- Jupyter notebook report

### Quickstart
```bash
cd LFW
# Run notebook: LFW_AVNISH_AGRAWAL_23BAI10628.ipynb
```

---

## 7. Lunar Lander RL Agent

**Solve LunarLander-v3 with Dueling DQN (custom PyTorch) and PPO (SB3)**

### Features
- Dueling DQN architecture (Value + Advantage streams)
- 8D state space, 4 discrete actions
- Stable-Baselines3 PPO baseline
- FastAPI inference API + Streamlit dashboard
- Single-file pipeline (`lunar_lander_pipeline.py`)

### Quickstart
```bash
cd "Lunar Lander RL agent"
pip install -r requirements.txt
python lunar_lander_pipeline.py --mode train_dqn
python lunar_lander_pipeline.py --mode train_ppo
python lunar_lander_pipeline.py --mode evaluate
python lunar_lander_pipeline.py --mode api    # Terminal 1
python lunar_lander_pipeline.py --mode ui     # Terminal 2
```

---

## 8. Movie Recommendation System

**Hybrid recommender: Content-based (TF-IDF) + Collaborative (SVD/PyTorch)**

### Features
- Modular architecture (`src/`, `app/`, `tests/`)
- Content engine: TF-IDF on genres
- Collaborative engine: SVD matrix factorization (PyTorch)
- Hybrid blending strategy
- FastAPI + Streamlit interfaces
- Unit tests with pytest

### Quickstart
```bash
cd "Movie recommendation system"
pip install -r requirements.txt
python main.py --mode train
python main.py --mode evaluate
uvicorn app.api:app --reload    # Terminal 1
streamlit run app/ui.py         # Terminal 2
pytest tests/
```

---

## 9. RAG Chatbot

**Enterprise-grade Retrieval-Augmented Generation with LangChain + ChromaDB**

### Features
- Document ingestion (PDF/TXT) → chunking → embeddings
- ChromaDB vector store with similarity search
- LangChain RAG chain with prompt templating
- Configurable LLM (FakeListLLM for testing, swap for OpenAI/HF)
- FastAPI + Streamlit interfaces
- Unit tests for ingestion and API

### Quickstart
```bash
cd "RAG Chatbot"
pip install -r requirements.txt
# Add documents to data/raw_documents/
python main.py --mode ingest
python main.py --mode serve    # Terminal 1
python main.py --mode ui       # Terminal 2
python main.py --mode test
```

---

## Common Requirements

Most projects share these core dependencies:
- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`
- `fastapi`, `uvicorn`, `pydantic`, `streamlit`
- `pytest` for testing

---

## Verification Tool

The repository comes with a root script to verify the structural integrity of all project folders:
```bash
python validate_projects.py
```

---

## License

MIT License - See individual project directories for details.