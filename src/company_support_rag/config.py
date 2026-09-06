import collections
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    chat_model: str = os.getenv("CHAT_MODEL", "granite4.1:3b")
    embed_model: str = os.getenv("EMBED_MODEL", "granite-embedding:278m")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./chroma_store"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    collections = {"hr": "hr_policy", "engineering": "engineering", "onboarding": "onboarding", "product":"productkb", "security":"security"}
    source_files = {"hr": "company_hr_policy.txt", "engineering": "engineering_standards.txt", "onboarding": "onboardng_guide.txt", "product":"product_knowledge_base.txt", "security":"security_policy.txt"}

settings = Settings()