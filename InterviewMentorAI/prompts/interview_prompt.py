def get_round_instructions(interview_round):
    instructions = {
        "HR Round": """
Focus on:
- communication
- teamwork
- leadership
- strengths
- weaknesses
- career goals
""",
        "Technical Round": """
Focus on:
- coding
- DSA
- debugging
- projects
- implementation details
""",
        "System Design Round": """
Focus on:
- scalability
- architecture
- APIs
- databases
- distributed systems
""",
        "Behavioral Round": """
Focus on:
- STAR method
- adaptability
- collaboration
- conflict resolution
""",
        "Aptitude Round": """
Focus on:
- logical reasoning
- quantitative aptitude
- analytical thinking
""",
    }
    return instructions.get(interview_round, "")


def build_interview_question_prompt(
    interview_round,
    round_instruction,
    resume_summary,
    previous_conversation,
    last_answer,
    context,
):
    return f"""
You are an expert interviewer.

Conduct a professional {interview_round} interview.

Round Instructions:
{round_instruction}

Resume Summary:
{resume_summary}

Previous Conversation:
{previous_conversation}

Previous Candidate Answer:
{last_answer}

Resume Context:
{context}

Rules:
- Ask ONLY ONE question
- Avoid repeating questions
- Generate realistic follow-up questions
- Focus on projects, academics and skills
- Keep questions short and clear
- First question should come from academics
- Ask intelligent follow-up questions

Interview Question:
"""
