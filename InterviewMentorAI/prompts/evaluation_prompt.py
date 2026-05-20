def build_evaluation_prompt(question, answer):
    return f"""
You are a senior technical interviewer.

Evaluate the candidate professionally.

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluate based on:
1. Technical Accuracy
2. Clarity
3. Depth
4. Communication
5. Confidence

STRICTLY provide:

Score: X/10
Technical Accuracy: X/10
Communication: X/10
Confidence: X/10

Then provide:
- Strengths
- Weaknesses
- Improvement Suggestions

Give structured professional feedback.
"""
