import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(files):
    all_docs = []
    processed_files = set()

    for file in files:
        if file.name in processed_files:
            continue

        processed_files.add(file.name)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                temp_path = tmp_file.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            os.remove(temp_path)

            for doc in docs:
                doc.metadata["source"] = file.name

            all_docs.extend(docs)

        except Exception:
            continue

    return all_docs
