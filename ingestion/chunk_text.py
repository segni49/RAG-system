# ingestion/chunk_text.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_and_clean import load_all_pdfs, clean_text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    raw_text = load_all_pdfs("data/")
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    print(f"✅ Total Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{chunk}\n")