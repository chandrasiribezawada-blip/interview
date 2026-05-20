from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)
