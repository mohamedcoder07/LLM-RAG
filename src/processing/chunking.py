from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import select_docs








def doc_splitting():
    print("CHUNKING...")
    relevant_docs = select_docs()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 0)
    
    for doc in relevant_docs:
        texts = text_splitter.split_text(relevant_docs.page_content)
    return texts