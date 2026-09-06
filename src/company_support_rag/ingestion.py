from fileinput import filename
from importlib import metadata
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core import documents
from langchain_core.documents import Document
from .config import settings
from .models import get_embeddings

def chunk_by_paragraph(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")
    return [Document(page_content=p.strip(), metadata={"source": str(file_path)})
            for p in text.split("\n\n") if len(p.split) >= 50 and not p.strip().startswith("===")]

def build_collection(file_path: Path, collection_name: str) -> int:
    documents = chunk_by_paragraph(file_path)
    if not documents: raise ValueError(f"No usable paragraphs in {file_path}")
    Chroma.from_documents(documents=documents, embedding=get_embeddings(), collection_name=collection_name, persist_directory=settings.chroma_dir)
    return len(documents)

def ingest_all() -> dict[str, int]:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return {intent: build_collection(settings.data_dir/filename, settings.collections[intent])
            for intent, filename in settings.source_files.items()}