import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env", override=False)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
FAISS_INDEX_PATH = str(BASE_DIR / "faiss_index")
APP_TITLE = "🎯 InterviewMentor AI – Multi-Round Mock Interview Assistant"
