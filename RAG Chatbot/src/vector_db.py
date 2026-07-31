
from langchain_community.vectorstores import Chroma
# Using FakeEmbeddings for out-of-the-box execution without API Keys
# Swap to OpenAIEmbeddings for production
from langchain_community.embeddings import FakeEmbeddings

def get_vector_store(persist_directory):
    embeddings = FakeEmbeddings(size=384)
    vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    return vector_store

def create_and_persist_vector_store(docs, persist_directory):
    embeddings = FakeEmbeddings(size=384)
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    vector_store.persist()
    return vector_store
