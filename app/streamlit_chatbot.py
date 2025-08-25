# app/streamlit_chatbot.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from generation.generate_answer import build_chain, generate_answer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)

# ✅ Load FAISS vectorstore
vectorstore = FAISS.load_local(
    "rag_project/vectorstore",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ✅ Build HuggingFaceHub chain
chain = build_chain()

# ✅ Streamlit UI
st.set_page_config(page_title="Naive RAG Chatbot", layout="centered")
st.title("🧠 Naive RAG Chatbot")
st.markdown("Ask a question based on your documents. Powered by HuggingFaceHub (`flan-t5-base`).")

query = st.text_input("🔍 Your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = generate_answer(vectorstore, query, chain)
            st.markdown(f"**🧠 Answer:**\n\n{answer}")
        except Exception as e:
            st.error(f"❌ Error: {e}")