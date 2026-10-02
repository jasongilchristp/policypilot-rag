from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from .config import settings


@lru_cache(maxsize=None)
def get_chat_model():
    return init_chat_model(
        model=settings.chat_model,
        model_provider=settings.model_provider,
        temperature=0
    )


@lru_cache(maxsize=None)
def get_intent_model():
    return init_chat_model(
        model=settings.intent_model,
        model_provider=settings.model_provider,
        temperature=0
    )


@lru_cache(maxsize=None)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=settings.embed_model)