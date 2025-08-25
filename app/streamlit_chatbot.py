# app/streamlit_chatbot.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from generation.generate_answer import build_chain, generate_answer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Use a lighter embedding model to avoid rate limits
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ✅ Load pre-generated FAISS index
vectorstore = FAISS.load_local(
    "rag_project/vectorstore",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Build chain
chain = build_chain()

# Streamlit UI
st.set_page_config(page_title="Naive RAG Chatbot", layout="centered")
st.title("🧠 Naive RAG Chatbot")
st.markdown("Ask a question based on your documents. Powered by HuggingFace (`flan-t5-base`).")

query = st.text_input("🔍 Your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = generate_answer(vectorstore, query, chain)
            st.markdown(f"**🧠 Answer:**\n\n{answer}")
        except Exception as e:
            st.error(f"❌ Error: {e}")