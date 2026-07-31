# Movie Recommendation System

## Overview
This project implements a hybrid movie recommendation engine that combines content-based filtering (TF-IDF) with collaborative filtering (SVD) to provide personalized recommendations addressing cold-start and data sparsity challenges.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python main.py --mode train`
3. Evaluate performance: `python main.py --mode evaluate`
4. Run API: `uvicorn app.api:app`
5. Launch UI: `streamlit run app/ui.py`

## Directory Structure
- `app/`: Application components including API and UI
- `data/`: Raw and processed data storage
- `models/`: Trained model artifacts
- `src/`: Source code modules for recommendation logic

## Usage
Execute training to build the recommendation model, evaluate to assess accuracy, then deploy using either the FastAPI backend or Streamlit UI for user interaction.

## Contributing
Contributions are welcome. Please follow standard coding practices and submit improvements via pull requests.

## License
MIT License