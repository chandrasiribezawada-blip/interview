from langchain_groq import ChatGroq


def create_llm(api_key, model_name, temperature=0.3):
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Set GROQ_API_KEY in InterviewMentorAI/.env or the root .env."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )
