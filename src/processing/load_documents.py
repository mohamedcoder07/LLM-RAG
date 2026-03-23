from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader






def load_docs():
    print("LOADING DOCS...")

    docs_folder = Path.cwd().parent / "datasets"
    docs_path = sorted(Path(docs_folder).glob("*.pdf"))
    
    for path in docs_path :    
        loader = PyMuPDFLoader(file_path=path)
        docs = loader.load()    
    return docs



def select_docs():
    docs = load_docs()
    

    return docs



# def doc_processing(args):
#     return