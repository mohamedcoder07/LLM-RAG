import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)


sys.path.append("C:\\Users\\user\\Desktop\\LLM-RAG\\src")

from processing.load_documents import load_docs#, select_docs, process_docs
from processing.chunking import get_chunks
from processing.embedder import embedding_model
from retrieval.vector_store import get_vector_store, get_retriever
from generation.generator import generate_answer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


warnings.filterwarnings("ignore")


def main():

    folder_path = Path.cwd()/ "data"/"pdf"#.parent.parent 
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    question = input("Quelle est votre question ?\n-> ")
    print("")
    relevant_docs = load_docs(folder_path)
    chunks = get_chunks(relevant_docs)
    model = embedding_model(embed_model)            
    vector_store = get_vector_store(chunks, model)   
    retriever = get_retriever(vector_store)
    print("-" * 50)    
    print("[RAG] PIPELINE READY")
    print("-" * 50)    
    answer = generate_answer(question, retriever, None)
    print(f"Response :\n-> {answer}\n")

if __name__ == "__main__":
    main()