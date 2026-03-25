import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]='0'

from langchain_openai import OpenAI
from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def call_llm(model_type = "llama"):    
    if model_type == "gpt":
        llm = OpenAI()
    elif model_type == "llama":
        llm = "llama3.2:3b"


    return llm

def get_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(question, retriever, model_type):
    
    # llm = call_llm(model_type)
    llm = ChatOllama(model= model_type) 
    # You need to check if you have ollama installed on your laptop 
    # and pull the model you want using "ollama pull model_name"
    
    prompt = """Tu es un assistant financier expert. 
    Utilise UNIQUEMENT le contexte ci-dessous pour répondre.
    Si la réponse n'est pas dans le contexte, dis "Je ne trouve pas cette information dans le rapport."
    Ne invente rien.

    context: {context}

    question: {query}

    answer: """

    follow_instruction = PromptTemplate.from_template(prompt)

    rag_chain = (
        {"context": retriever | get_context, "query": RunnablePassthrough()}
        | follow_instruction
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)

    return answer