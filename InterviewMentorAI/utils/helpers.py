import hashlib


def unique_documents(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)

    return unique_docs
