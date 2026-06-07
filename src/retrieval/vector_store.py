import os
from pathlib import Path

from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"

# There is a difference : vector_store != vector_database !!
# Here we are using the vector store -FAISS-
def _create_vector_store(chunks, model, save_embeddings = False):
    vector_store = FAISS.from_documents(chunks, model)
    
    if save_embeddings:
        vector_store.save_local("faiss_index")
        
    return vector_store


def get_vector_store(chunks, model):
    if Path(INDEX_DIR).exists():
        vector_store = FAISS.load_local(INDEX_DIR, model)
    else:
        print("[Step 3] VECTOR REPRESENTATION & STORING")            
        vector_store = _create_vector_store(chunks, 
                                            model, 
                                            save_embeddings = False)
    
    return vector_store

def get_retriever(vector_store: FAISS):
    print("[Step 4] RETRIEVING RELEVENT DOCUMENTS")     
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    return retriever 

