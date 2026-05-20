from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from ..retrieval.retriever import build_retriever


def build_qa_chain(llm, vectorstore, interview_round, top_k):
    retriever = build_retriever(vectorstore, top_k)

    prompt_template = f"""
You are InterviewMentor AI.

You are conducting a {interview_round} interview.

Rules:
- Answer professionally
- Ask interview-style responses
- Use retrieved context
- If no answer found say:
"I could not find enough information."

Context:
{{context}}

Question:
{{question}}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain
