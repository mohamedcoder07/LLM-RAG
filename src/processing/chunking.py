from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_chunks(documents):
    print("CHUNKING...")                       
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
    
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.page_content))
        
    return chunks