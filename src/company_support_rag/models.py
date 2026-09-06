from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from .config import settings

def get_llm():
    return init_chat_model(f"ollama:{settings.chat_model}")

def get_embeddings():
    return OllamaEmbeddings(model=settings.embed_model)