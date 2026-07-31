
import streamlit as st
import requests

st.title("RAG Chatbot Capstone UI")
query = st.text_input("Ask a question based on uploaded documents:")
if st.button("Submit"):
    try:
        res = requests.post("http://localhost:8000/query", json={"query": query})
        if res.status_code == 200:
            data = res.json()
            st.success(data["answer"])
            st.info(f"Sources: {data['sources']}")
        else:
            st.error("API Error")
    except:
        st.error("Could not connect to API. Run with `--mode serve`.")
