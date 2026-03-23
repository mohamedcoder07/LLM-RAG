from langchain_community.embeddings import HuggingFaceEmbeddings


def embedding_model(model_id):
    model = HuggingFaceEmbeddings(
        model_name=model_id,
        # multi_process=True,
        model_kwargs={"device": "cpu"},  # or "cuda"
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )    
    
    return model













# Another way to proceed is done below without using the package langchain

# import requests

# model_id = "sentence-transformers/all-MiniLM-L6-v2"
# hf_token = "your-hf-token"

# api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}/pipeline/feature-extraction"
# headers = {"Authorization": f"Bearer {hf_token}"}

# def get_embedding(texts):
#     response = requests.post(api_url, headers=headers, json={"inputs": texts}) #, "options":{"wait_for_model":True}
#     if response.status_code == 200: 
#         embed = np.array(response.json())
#         embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
#         return emb.tolist()
#     else:
#         print("ERROR:", response.text)
#         return None