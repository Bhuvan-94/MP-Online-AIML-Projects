
from src.document_loader import load_and_split_documents
import os

def test_document_loader(tmp_path):
    test_dir = tmp_path / "data"
    test_dir.mkdir()
    p = test_dir / "doc.txt"
    p.write_text("This is test data for LangChain RAG pipeline.")
    
    docs = load_and_split_documents(str(test_dir))
    assert len(docs) > 0
    assert "test data" in docs[0].page_content
