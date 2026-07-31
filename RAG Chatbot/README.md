# RAG Chatbot

## Overview
This capstone project implements an enterprise-grade Retrieval-Augmented Generation (RAG) system that processes custom documents, indexes them in a vector database, and generates augmented LLM responses through a streamlined pipeline.

## Architecture
- **Document Ingestion**: Parse and load custom PDF/TXT files
- **Text Processing**: Split content into manageable chunks
- **Vector Embedding**: Convert text into embeddings using LangChain
- **Vector Storage**: Index embeddings in ChromaDB for similarity search
- **Response Generation**: Use LLMs to generate context-aware answers
- **Interface**: Streamlit UI for user interaction

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Add documents to `data/raw_documents/`
3. Run ingestion pipeline: `python main.py --mode ingest`
4. Launch API server: `python main.py --mode serve`
5. Start UI: `python main.py --mode ui`
6. Execute tests: `python main.py --mode test`

## Directory Structure
- `app/`: API and UI components
- `data/`: Raw documents and processed vectors
- `models/`: Trained models and embeddings
- `src/`: Core RAG pipeline implementation

## Usage
Follow the quickstart sequence to process documents, load vector store, and serve responses via API or UI.

## Contributing
Contributions are welcome. Maintain code quality standards and submit pull requests for enhancements.

## License
MIT License