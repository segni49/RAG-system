# ingestion/load_and_clean.py

import os
import re
from PyPDF2 import PdfReader

def load_all_pdfs(folder_path: str) -> str:
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            reader = PdfReader(full_path)
            for page in reader.pages:
                all_text += page.extract_text() + "\n\n"
    if not all_text:
        raise ValueError("No PDF content found in the folder.")
    return all_text

def clean_text(text: str) -> str:
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"(Disclaimer|Confidential)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n(?=\w)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def preview_text(text: str, length: int = 1000) -> None:
    print("\n🧼 Cleaned Text Preview:\n")
    print(text[:length])
    print("\n--- End of Preview ---\n")

if __name__ == "__main__":
    raw_text = load_all_pdfs("data/")
    cleaned_text = clean_text(raw_text)
    preview_text(cleaned_text)