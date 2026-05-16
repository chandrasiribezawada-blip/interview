# app.py

import streamlit as st
import os
import time
import tempfile
import hashlib
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import RetrievalQA

# ==========================================
# LOAD ENV VARIABLES (API KEY HIDDEN)
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="InterviewMentor AI",
    layout="wide"
)

st.title("🎯 InterviewMentor AI – Multi-Round Mock Interview Assistant")


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "score" not in st.session_state:
    st.session_state.score = 0

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# NEW STATES

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "memory" not in st.session_state:

    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "question_number" not in st.session_state:
    st.session_state.question_number = 0
if "interview_history" not in st.session_state:
    st.session_state.interview_history = []

if "total_score" not in st.session_state:
    st.session_state.total_score = 0

if "max_questions" not in st.session_state:
    st.session_state.max_questions = 25

if "interview_completed" not in st.session_state:
    st.session_state.interview_completed = False
    
if "resume_summary" not in st.session_state:
    st.session_state.resume_summary = ""
# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.header("⚙ Interview Configuration")

interview_round = st.sidebar.selectbox(
    "Select Interview Round",
    [
        "HR Round",
        "Technical Round",
        "System Design Round",
        "Behavioral Round",
        "Aptitude Round"
    ]
)

top_k = st.sidebar.slider(
    "Top-K Retrieval",
    1,
    10,
    4
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    200,
    2000,
    1000
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0,
    500,
    200
)

st.sidebar.slider(
    "Number of Questions",
    5,
    10,
    25,
    key="max_questions"
)
# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "📄 Upload Resume / Notes / PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embedding_model()
# ==========================================
# LOAD EXISTING VECTORSTORE
# ==========================================

if os.path.exists("faiss_index"):

    try:

        st.session_state.vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    except:

        pass
# ==========================================
# REMOVE DUPLICATE DOCUMENTS
# ==========================================

def unique_documents(docs):

    seen = set()

    unique_docs = []

    for doc in docs:

        content_hash = hashlib.md5(
            doc.page_content.encode()
        ).hexdigest()

        if content_hash not in seen:

            seen.add(content_hash)

            unique_docs.append(doc)

    return unique_docs
# ==========================================
# LOAD GROQ MODEL
# ==========================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

llm = load_llm()


# ==========================================
# PROCESS DOCUMENTS
# ==========================================

# ==========================================
# PROCESS DOCUMENTS
# ==========================================

def process_documents(files):

    if chunk_overlap >= chunk_size:

        st.error(
            "Chunk overlap must be smaller than chunk size"
        )

        st.stop()

    all_docs = []

    processed_files = set()

    for file in files:

        if file.name in processed_files:

            st.warning(
                f"Skipping duplicate file: {file.name}"
            )

            continue

        processed_files.add(file.name)

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(file.read())

                temp_path = tmp_file.name

            loader = PyPDFLoader(temp_path)

            docs = loader.load()

            os.remove(temp_path)

            for doc in docs:

                doc.metadata["source"] = file.name

            all_docs.extend(docs)

        except Exception as e:

            st.error(
                f"Error processing {file.name}: {e}"
            )

            continue

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(all_docs)

    split_docs = unique_documents(split_docs)

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    # SAVE VECTOR DB

    vectorstore.save_local("faiss_index")

    return vectorstore
# ==========================================
# BUILD QA CHAIN
# ==========================================

def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": top_k,
        "fetch_k": 20,
        "lambda_mult": 0.7
        }
    )

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
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    return qa_chain

# ==========================================
# RESUME ANALYZER
# ==========================================

def analyze_resume():

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    docs = retriever.invoke(
        "Extract candidate skills, projects, education and technologies"
    )

    docs = unique_documents(docs)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

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

# ==========================================
# ROUND INSTRUCTIONS
# ==========================================

def get_round_instructions():

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
"""
    }

    return instructions.get(interview_round, "")

def get_default_question():

    defaults = {
        "HR Round": "Tell me about a time you handled a difficult situation at work.",
        "Technical Round": "Explain how you would solve a technical problem using best practices.",
        "System Design Round": "How would you design a scalable, fault-tolerant system for real-time data processing?",
        "Behavioral Round": "Describe a project where you had to adapt to a major change in requirements.",
        "Aptitude Round": "If five people can finish a task in three days, how many days would ten people need to finish the same task?"
    }

    return defaults.get(interview_round, "Tell me about your most important professional achievement.")


# ==========================================
# GENERATE INTERVIEW QUESTION
# ==========================================

# ==========================================
# GENERATE INTERVIEW QUESTION
# ==========================================

def generate_interview_question():

    retrieval_start = time.time()

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    docs = retriever.invoke(
        "Generate interview questions from candidate resume and skills"
    )

    retrieval_end = time.time()

    retrieval_latency = round(
        retrieval_end - retrieval_start,
        2
    )

    st.sidebar.info(
        f"Retrieval Latency: {retrieval_latency}s"
    )

    docs = unique_documents(docs)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    round_instruction = get_round_instructions()

    last_answer = ""

    if st.session_state.interview_history:

        last_answer = (
            st.session_state.interview_history[-1]["answer"]
        )

    chat_history = st.session_state.memory.load_memory_variables(
        {}
    )["chat_history"]

    prompt = f"""
You are an expert interviewer.

Conduct a professional {interview_round} interview.

Round Instructions:
{round_instruction}

Resume Summary:
{st.session_state.resume_summary}

Previous Conversation:
{chat_history}

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

    response = llm.invoke(prompt)

    return response.content
# ==========================================
# EVALUATE ANSWER
# ==========================================
# ==========================================
# EVALUATE ANSWER
# ==========================================

def evaluate_answer(question, answer):

    prompt = f"""
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

    response = llm.invoke(prompt)

    evaluation = response.content

    # ======================================
    # EXTRACT SCORE
    # ======================================

    score = 5

    try:

        first_line = evaluation.split("\n")[0]

        score = int(
            first_line.split(":")[1]
            .split("/")[0]
            .strip()
        )

    except:
        score = 5

    # ======================================
    # UPDATE TOTAL SCORE
    # ======================================

    st.session_state.total_score += score

    # ======================================
    # STORE MEMORY
    # ======================================

    st.session_state.memory.save_context(
        {"input": question},
        {"output": answer}
    )

    # ======================================
    # STORE INTERVIEW HISTORY
    # ======================================

    st.session_state.interview_history.append(
        {
            "question": question,
            "answer": answer,
            "evaluation": evaluation,
            "score": score
        }
    )

    return evaluation

# ==========================================
# BUILD KNOWLEDGE BASE
# ==========================================

if st.button("🚀 Build Interview Knowledge Base"):

    if not uploaded_files:

        st.warning("Please upload PDF files")
        st.stop()

    with st.spinner("Processing Documents..."):

        vectorstore = process_documents(uploaded_files)

        qa_chain = build_chain(vectorstore)

        st.session_state.vectorstore = vectorstore
        resume_summary = analyze_resume()

        st.session_state.resume_summary = resume_summary
        st.session_state.qa_chain = qa_chain

        st.success("✅ Knowledge Base Ready")
        st.subheader("📄 Resume Understanding")

        st.write(st.session_state.resume_summary)

# ==========================================
# MOCK INTERVIEW SECTION
# ==========================================

st.header("💬 AI Mock Interview")

# ==========================================
# START INTERVIEW
# ==========================================

if st.button("🎤 Start Interview"):

    if st.session_state.vectorstore is None:

        st.warning("Please build knowledge base first")
        st.stop()

    # RESET STATES

    st.session_state.question_number = 1
    st.session_state.total_score = 0
    st.session_state.interview_history = []
    st.session_state.interview_completed = False

    question = generate_interview_question()

    st.session_state.current_question = question
    st.session_state.interview_started = True

# ==========================================
# DISPLAY INTERVIEW HISTORY
# ==========================================

if st.session_state.interview_history:

    st.subheader("📜 Interview History")

    for idx, item in enumerate(
        st.session_state.interview_history,
        start=1
    ):

        with st.expander(f"Question {idx}"):

            st.markdown(
                f"### 🧠 Question\n{item['question']}"
            )

            st.markdown(
                f"### 👤 Your Answer\n{item['answer']}"
            )

            st.markdown(
                f"### 📊 Evaluation\n{item['evaluation']}"
            )

            st.success(
                f"Score: {item['score']}/10"
            )

# ==========================================
# ACTIVE INTERVIEW
# ==========================================

if (
    st.session_state.interview_started
    and not st.session_state.interview_completed
):

    st.subheader(
        f"🧠 Question "
        f"{st.session_state.question_number}/"
        f"{st.session_state.max_questions}"
    )

    st.info(st.session_state.current_question)

    user_answer = st.text_area(
        "Your Answer",
        key=f"answer_box_{st.session_state.question_number}"
    )

    if st.button("Submit Answer"):

        if not user_answer.strip():

            st.warning("Please enter your answer")
            st.stop()

        with st.spinner("Evaluating Answer..."):

            start_time = time.time()

            evaluation = evaluate_answer(
                st.session_state.current_question,
                user_answer
            )

            end_time = time.time()

            total_latency = round(
                end_time - start_time,
                2
            )

            st.markdown("## 📊 Evaluation")

            st.markdown(evaluation)

            st.info(
                f"⏱ Response Evaluation Time: "
                f"{total_latency} seconds"
            )

            # ==================================
            # NEXT QUESTION LOGIC
            # ==================================

            st.session_state.question_number += 1

            if (
                st.session_state.question_number >
                st.session_state.max_questions
            ):

                st.session_state.interview_completed = True

            else:

                next_question = generate_interview_question()

                st.session_state.current_question = next_question

            st.rerun()

# ==========================================
# FINAL REPORT
# ==========================================

if st.session_state.interview_completed:

    st.divider()

    st.header("🏁 Interview Completed")

    average_score = round(
        st.session_state.total_score /
        st.session_state.max_questions,
        2
    )

    st.success(
        f"🎯 Final Interview Score: "
        f"{average_score}/10"
    )

    # ======================================
    # PERFORMANCE FEEDBACK
    # ======================================

    if average_score >= 8:

        st.balloons()

        st.success(
            "Excellent performance!"
        )

    elif average_score >= 6:

        st.info(
            "Good performance with room for improvement."
        )

    else:

        st.warning(
            "Need more preparation."
        )

    # ======================================
    # FINAL SUMMARY
    # ======================================

    st.subheader("📌 Overall Interview Summary")

    strengths = []
    weaknesses = []

    for item in st.session_state.interview_history:

        evaluation_text = item["evaluation"]

        if "Strength" in evaluation_text:
            strengths.append("Good technical understanding")

        if "Weak" in evaluation_text:
            weaknesses.append("Need deeper explanations")

    st.markdown("### ✅ Strengths")
    st.write(
        list(set(strengths))
        if strengths
        else ["Good participation"]
    )

    st.markdown("### ⚠ Areas to Improve")
    st.write(
        list(set(weaknesses))
        if weaknesses
        else ["Improve technical depth"]
    )

