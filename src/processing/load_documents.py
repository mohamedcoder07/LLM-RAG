from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


def load_docs(folder_path):
    print("LOADING DOCS...")    
    docs_path = sorted(Path(folder_path).glob("*.pdf"))
    documents = []

    for path in docs_path :    
        loader = PyMuPDFLoader(file_path=path)
        doc_loaded = loader.load()    
        documents.extend(doc_loaded) 

    return documents



def select_docs():
    docs = load_docs()
    # Not implemented yet / In order to select the relevant docs

    return docs


