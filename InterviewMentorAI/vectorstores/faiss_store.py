from pathlib import Path
from langchain_community.vectorstores import FAISS


def create_faiss_store(split_docs, embeddings):
    return FAISS.from_documents(split_docs, embeddings)


def load_faiss_store(embeddings, index_path):
    if Path(index_path).exists():
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None
