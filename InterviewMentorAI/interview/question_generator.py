from ..prompts.interview_prompt import (
    build_interview_question_prompt,
    get_round_instructions,
)
from ..retrieval.retriever import build_retriever
from ..utils.helpers import unique_documents


def generate_interview_question(
    llm,
    vectorstore,
    interview_round,
    resume_summary,
    memory,
    top_k,
):
    retriever = build_retriever(vectorstore, top_k)

    docs = retriever.invoke(
        "Generate interview questions from candidate resume and skills"
    )

    docs = unique_documents(docs)
    context = "\n".join([doc.page_content for doc in docs])
    round_instruction = get_round_instructions(interview_round)

    last_answer = ""
    if memory:
        stored = memory.load_memory_variables({}).get("chat_history", "")
        last_answer = stored

    prompt = build_interview_question_prompt(
        interview_round=interview_round,
        round_instruction=round_instruction,
        resume_summary=resume_summary,
        previous_conversation=last_answer,
        last_answer=last_answer,
        context=context,
    )

    response = llm.invoke(prompt)
    return response.content
