# ingestion/embed_store.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from load_and_clean import load_all_pdfs, clean_text
from chunk_text import chunk_text
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def embed_and_store(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    vectorstore.save_local("rag_project/vectorstore")
    print("✅ Vectorstore saved to rag_project/vectorstore")

if __name__ == "__main__":
    raw_text = load_all_pdfs("data/")
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    embed_and_store(chunks)