import os
import pickle
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document


CACHE_CHUNKS_FILE = "data/chunks/chunks_cache.pkl"

def get_chunks(documents):
    print("[Step 2] CHUNKING") 
    
    # Check if chunks are already cached to save processing time
    if Path(CACHE_CHUNKS_FILE).exists():
        with open(CACHE_CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
            print(f"[INFO] Nombre de chunks chargés depuis le cache : {len(chunks)}")
            return chunks
        
    chunks = []

    # Define markdown headers to split the text by structure
    headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    # Initialize the splitter (keeping headers in the final chunk text)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # On garde le titre dans le texte !
    )

    # Process each document and collect the resulting chunks
    for doc in documents:        
        md_chunks = markdown_splitter.split_text(doc.page_content)
        chunks.extend(md_chunks)    

    print(f"[INFO] Nombre de chunks créés : {len(chunks)}")

    # Save the newly created chunks to cache for future runs
    with open(CACHE_CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return chunks
