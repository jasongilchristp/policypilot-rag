from langchain_chroma import Chroma
from .config import settings
from .models import get_embeddings

def retrieve_from_chroma(query: str, collection_name: str, k: int | None= None):
    store = Chroma(collection_name=collection_name, embedding_function=get_embeddings, persist_directory=settings.chroma_dir)
    return store.similarity_search(query, k=k or settings.retrieval_k)

def format_context(documents):
    return "\n\n".join(f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in documents)