# 🧠 Naive RAG Chatbot - Ask about Agriculture

A lightweight, local-first Retrieval-Augmented Generation (RAG) chatbot built with LangChain, FAISS, HuggingFace embeddings, and Ollama-powered LLMs. It answers questions based on your own PDF documents — and gracefully says "I don't know" when asked something outside its context.

---

## 🚀 Features

- ✅ Loads and cleans multiple PDFs using PyPDF2
- ✅ Splits text into overlapping chunks for semantic search
- ✅ Embeds chunks using HuggingFace sentence transformers
- ✅ Stores embeddings in a FAISS vectorstore
- ✅ Answers questions using a local LLM (e.g., `tinyllama` via Ollama)
- ✅ Grounded generation: refuses to hallucinate answers
- ✅ Streamlit UI for interactive querying

---

## 📁 Project Structure

```text
RAG-system/
├── app/
│   └── streamlit_chatbot.py         # Streamlit frontend
├── generation/
│   └── generate_answer.py           # RAG logic and LLM prompt
├── ingestion/
│   ├── load_and_clean.py            # PDF loading and cleaning
│   ├── chunk_text.py                # Text chunking
│   └── embed_store.py               # Embedding and FAISS storage
├── retrieval/
│   └── retrieve_chunks.py           # Chunk retrieval for testing
├── data/
│   └── *.pdf                        # Your source documents
├── rag_project/
│   └── vectorstore/                 # FAISS index and metadata
└── requirements.txt                 # Dependencies
```

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Usage

### Step 1: Prepare your PDFs

Place all your `.pdf` files inside the `data/` folder.

### Step 2: Generate the vectorstore

```bash
python ingestion/embed_store.py
```

This will create `rag_project/vectorstore/index.faiss` and `index.pkl`.

### Step 3: Launch the chatbot

```bash
streamlit run app/streamlit_chatbot.py
```

Ask questions based on your documents. If the answer isn’t found, the bot will say:

> “I don’t know. That’s outside the context of the documents I was trained on.”

---

## 🧠 Model Notes

- Embedding model: `sentence-transformers/paraphrase-MiniLM-L3-v2`
- LLM: `tinyllama` via [Ollama](https://ollama.com)
- Vectorstore: FAISS (local, fast, scalable)

---

## 🧪 Testing Retrieval

You can test chunk retrieval directly:

```bash
python retrieval/retrieve_chunks.py
```

---

## 📦 Deployment

This project is designed to run locally. For deployment to HuggingFace Spaces or Streamlit Cloud:

- Include `requirements.txt`
- Add a `README.md`
- Ensure FAISS index is pre-generated
- Use CPU-friendly models

---

## 🙋 FAQ

**Q: What happens if I ask something outside the PDFs?**  
A: The chatbot will respond with “I don’t know…” — no hallucination.

**Q: Can I use scanned PDFs?**  
A: PyPDF2 only works with text-based PDFs. For scanned documents, consider switching to `pdfplumber` or `UnstructuredPDFLoader`.

**Q: Can I use a different LLM?**  
A: Yes! Just change the model name in `generate_answer.py`.

---

## 📄 License

MIT License — free to use, modify, and share.

---

## ✨ Credits

Built by [Segni Abera](https://github.com/segni49)  
Powered by LangChain, HuggingFace, FAISS, and Ollama
