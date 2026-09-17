from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from .config import settings

def get_llm():
    return init_chat_model(
        model=settings.chat_model,
        model_provider="ollama",
        temperature=0
    )


def get_embeddings():
    return OllamaEmbeddings(model=settings.embed_model)