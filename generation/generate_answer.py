# generation/generate_answer.py

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint  # ✅ Correct import
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List
import streamlit as st

def format_context(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question using only the context below.
If the answer is not in the context, say "I don't know. That’s outside the context of the documents I was trained on."

Context:
{context}

Question:
{question}
"""
    )

def build_chain() -> RunnableSequence:
    prompt = build_prompt()
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        huggingfacehub_api_token=st.secrets["huggingface"]["token"],
        temperature=0.3,
        max_new_tokens=512
    )
    return prompt | llm

def generate_answer(vectorstore: FAISS, query: str, chain: RunnableSequence, k: int = 3) -> str:
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = format_context(retrieved_docs)
    return chain.invoke({"context": context, "question": query})