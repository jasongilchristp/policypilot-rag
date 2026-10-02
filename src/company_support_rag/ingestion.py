from fileinput import filename
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .config import settings
from .models import get_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_by_paragraph(file_path: Path) -> list[Document]:
    text = file_path.read_text(encoding="utf-8")
    recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # Maximum characters per chunk
            chunk_overlap=200,     # Overlap between chunks (20%)
            add_start_index=True   # Retains character metadata position
        )
    raw_chunks = recursive_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in raw_chunks]

def build_collection(file_path: Path, collection_name: str) -> int:
    documents = chunk_by_paragraph(file_path)
    if not documents: raise ValueError(f"No usable paragraphs in {file_path}")
    Chroma.from_documents(documents=documents, embedding=get_embeddings(), collection_name=collection_name, persist_directory=str(settings.chroma_dir))
    return len(documents)

def ingest_all() -> dict[str, int]:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return {intent: build_collection(settings.data_dir/filename, settings.collections[intent])
            for intent, filename in settings.source_files.items()}