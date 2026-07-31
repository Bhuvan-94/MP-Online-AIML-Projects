
import argparse
import sys
import subprocess
import uvicorn
from src.document_loader import load_and_split_documents
from src.vector_db import create_and_persist_vector_store

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ingest', 'serve', 'test', 'ui'], required=True)
    args, _ = parser.parse_known_args()
    
    if args.mode == 'ingest':
        print("Ingesting documents from data/raw_documents...")
        docs = load_and_split_documents('data/raw_documents')
        print(f"Loaded {len(docs)} chunks. Indexing into ChromaDB...")
        create_and_persist_vector_store(docs, 'data/vector_store')
        print("Ingestion complete!")
    elif args.mode == 'serve':
        from app.api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.mode == 'test':
        subprocess.run([sys.executable, "-m", "pytest"])
    elif args.mode == 'ui':
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui.py"])
