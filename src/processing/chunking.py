from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_chunks(documents):
    print("CHUNKING...")                       
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 150)
    
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.page_content))
    
    print(f"Nombre de chunks : {len(chunks)}")

    return chunks