import sys
sys.path.append("C:\\Users\\user\\Desktop\\LLM-RAG\\src")

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'

import warnings
warnings.filterwarnings(action="ignore")

from pathlib import Path

from processing.load_documents import load_docs, select_docs
from processing.chunking import get_chunks
from processing.embedder import embedding_model

from retrieval.vector_store import get_vector_store, get_retriever

from generation.generator import generate_answer

def main():
    folder_path = Path.cwd()/ "datasets"#.parent.parent 
    # print(folder_path)
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    question = input("Quelle est votre question ?\n-> ")
    print("")
    generation_model = "mistral:7b"

    relevant_docs = load_docs(folder_path)
    chunks = get_chunks(relevant_docs)
    model = embedding_model(embed_model)
    vector_store = get_vector_store(chunks, model)
    retriever = get_retriever(vector_store)
    answer = generate_answer(question, retriever, generation_model)
    print(f"Response :\n->{answer}\n")

if __name__ == "__main__":
    main()