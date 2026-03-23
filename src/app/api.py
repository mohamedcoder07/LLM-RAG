from pathlib import Path

from processing.load_documents import load_docs, select_docs
from processing.chunking import get_chunks
from processing.vector_store import get_vector_store
from processing.embedder import embedding_model

folder_path = Path.cwd().parent.parent / "datasets"
model_id = "sentence-transformers/all-MiniLM-L6-v2"

relevant_docs = load_docs(folder_path) 
chunks = get_chunks(relevant_docs)
model = embedding_model(model_id)
vector_store = get_vector_store(chunks, model)







