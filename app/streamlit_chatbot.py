# app/streamlit_chatbot.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from generation.generate_answer import build_chain, generate_answer
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(
    "rag_project/vectorstore",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

llm_chain = build_chain(model_name="tinyllama", temperature=0.3)

st.set_page_config(page_title="Naive RAG Chatbot", layout="centered")
st.title("🧠RAG Chatbot - Ask Questions  about ethiopian agriculture")
st.markdown("Ask a question about agriculture. Powered by local LLM (tinyllama).")

query = st.text_input("🔍 Your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = generate_answer(vectorstore, query, llm_chain)
            st.markdown(f"**🧠 Answer:**\n\n{answer}")
        except Exception as e:
            st.error(f"❌ Error: {e}")