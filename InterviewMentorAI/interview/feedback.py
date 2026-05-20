def summarize_interview_history(interview_history):
    strengths = []
    weaknesses = []

    for item in interview_history:
        evaluation_text = item.get("evaluation", "")
        if "Strength" in evaluation_text:
            strengths.append("Good technical understanding")
        if "Weak" in evaluation_text:
            weaknesses.append("Need deeper explanations")

    return list(set(strengths)) or ["Good participation"], list(set(weaknesses)) or ["Improve technical depth"]
