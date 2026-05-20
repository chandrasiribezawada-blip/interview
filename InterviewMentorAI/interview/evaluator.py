from ..prompts.evaluation_prompt import build_evaluation_prompt
from ..retrieval.retriever import build_basic_retriever
from ..utils.helpers import unique_documents


def analyze_resume(llm, vectorstore):
    retriever = build_basic_retriever(vectorstore, k=10)
    docs = retriever.invoke(
        "Extract candidate skills, projects, education and technologies"
    )
    docs = unique_documents(docs)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Analyze this resume professionally.

Extract:
1. Candidate Name
2. Education
3. Skills
4. Projects
5. Technologies
6. Experience Level

Resume:
{context}

Return concise structured summary.
"""

    response = llm.invoke(prompt)
    return response.content


def extract_score(evaluation_text):
    try:
        first_line = evaluation_text.split("\n")[0]
        return int(first_line.split(":")[1].split("/")[0].strip())
    except Exception:
        return 5


def evaluate_answer(llm, question, answer, memory, interview_history):
    prompt = build_evaluation_prompt(question, answer)
    response = llm.invoke(prompt)
    evaluation = response.content
    score = extract_score(evaluation)

    if memory:
        memory.save_context({"input": question}, {"output": answer})

    interview_history.append(
        {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "score": score,
        }
    )

    return evaluation, score
