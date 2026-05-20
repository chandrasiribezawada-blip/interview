def build_retriever(vectorstore, top_k):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )


def build_basic_retriever(vectorstore, k=10):
    return vectorstore.as_retriever(search_kwargs={"k": k})
