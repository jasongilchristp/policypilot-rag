import collections
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    chat_model: str = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
    intent_model: str = os.getenv("INTENT_MODEL", "openai/gpt-oss-20b")
    model_provider:  str = os.getenv("MODEL_PROVIDER", "groq")
    embed_model: str = os.getenv("EMBED_MODEL", "gemini-embedding-001")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./chroma_store"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    collections = {"hr": "hr_policy", "engineering": "engineering", "onboarding": "onboarding", "product":"productkb", "security":"security"}
    source_files = {"hr": "company_hr_policy.txt", "engineering": "engineering_standards.txt", "onboarding": "onboarding_guide.txt", "product":"product_knowledge_base.txt", "security":"security_policy.txt"}

settings = Settings()