# generation/generate_answer.py

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List

def format_context(docs: List[Document]) -> str:
    """
    Combines retrieved documents into a single context string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def build_prompt() -> PromptTemplate:
    """
    Creates a prompt that forces the model to stay grounded in context.
    """
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

def build_chain(model_name: str = "tinyllama", temperature: float = 0.3) -> LLMChain:
    """
    Constructs an LLMChain using a local Ollama model and the grounded prompt.
    """
    llm = ChatOllama(model=model_name, temperature=temperature)
    prompt = build_prompt()
    return LLMChain(llm=llm, prompt=prompt)

def generate_answer(vectorstore: FAISS, query: str, llm_chain: LLMChain, k: int = 3) -> str:
    """
    Retrieves context and generates an answer using the LLM chain.
    """
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    context = format_context(retrieved_docs)
    return llm_chain.run({"context": context, "question": query})

if __name__ == "__main__":
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

    query = "What is backpropagation in neural networks?"
    answer = generate_answer(vectorstore, query, llm_chain)

    print(f"\n🧠 Answer:\n{answer}")