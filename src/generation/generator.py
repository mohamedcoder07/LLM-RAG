import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import InferenceClient

from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq


def get_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def generate_answer(question, retriever, model_type=None):

    # Extraction dynamique du contexte depuis le vector store (retriever)
    docs = retriever.invoke(question) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(question)
    
    # # This part was just for debugging
    # print("\n=== DEBUG : DOCUMENTS RETROUVÉS PAR LE RAG ===")    
    # if not docs:
    #     print("Aucun document trouvé pour cette question !")
    # else:
    #     for i, doc in enumerate(docs):
    #         if doc.metadata.get('page', 'Inconnue')==12:
    #             print(f"\n[Document {i+1} | Source: {doc.metadata.get('source', 'Inconnue')} | Page: {doc.metadata.get('page', 'Inconnue')}]")
    #             print(f"Type: {doc.metadata.get('doc_type', 'Non spécifié')}")
    #             # On affiche les 500 premiers caractères pour voir le contenu sans inonder la console
    #             print(f"Contenu : \n{doc.page_content[:7000]}")
    #             print("...")
    # print("\n==============================================\n")


    # Fusion des contenus de toutes les pages trouvées
    context_text = get_context(docs)
    
    # Remplacement des variables {context} et {question} via un f-string
    instructions = (
        "Tu es un assistant financier expert et rigoureux.\n"
        "Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question.\n"
        "Sois extrêmement précis avec les chiffres et les tableaux.\n"
        "Si la réponse n'est pas dans le contexte, dis simplement : 'Je ne trouve pas cette information dans le rapport.'\n"
        "Ne devine rien, n'estime rien et n'invente rien de ta part."
    )

    # # Call LLM via Groq
    # raw_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.01,
        max_tokens=512
    )
    
    # # Prompt respecting the documentation including the "system" and "human" components
    messages = [
        ("system", instructions),
        ("human", f"--- CONTEXTE BIEN REEL ---\n{context_text}\n-------------------------\n\nQuestion: {question}\n\nRéponse :")
    ]


    response = llm.invoke(messages)

    return response.content    


    # # Call LLM from Hugging Face
    # hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # client = InferenceClient(
    #     model="Qwen/Qwen2.5-72B-Instruct", 
    #     token=hf_token
    # )

    # prompt = (
    #     f"{instructions}\n\n"
    #     f"--- CONTEXTE BIEN REEL ---\n{context_text}\n-------------------------\n\n"
    #     f"Question: {question}\n\n"
    #     "Réponse :"
    # )
    
    # # Envoi du prompt correctement formaté
    # response = client.chat_completion(
    #     messages=[{"role": "user", "content": prompt}],
    #     max_tokens=512,
    #     temperature=0.01  # Rigueur maximale pour les chiffres
    # )
    # return response.choices[0].message.content