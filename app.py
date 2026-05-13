# app.py

import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


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

def process_documents(files):

    all_docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

            tmp_file.write(file.read())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        # Add Metadata

        for doc in docs:

            doc.metadata["source"] = file.name

        all_docs.extend(docs)

    # ======================================
    # TEXT SPLITTING
    # ======================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(all_docs)

    # ======================================
    # VECTOR STORE
    # ======================================

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    return vectorstore


# ==========================================
# BUILD QA CHAIN
# ==========================================

def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
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


def get_default_question():

    defaults = {
        "HR Round": "Tell me about a time you handled a difficult situation at work.",
        "Technical Round": "Explain how you would solve a technical problem using best practices.",
        "System Design Round": "How would you design a scalable, fault-tolerant system for real-time data processing?",
        "Behavioral Round": "Describe a project where you had to adapt to a major change in requirements.",
        "Aptitude Round": "If five people can finish a task in three days, how many days would ten people need to finish the same task?"
    }

    return defaults.get(interview_round, "Tell me about your most important professional achievement.")


def generate_interview_question():

    prompt = f"""
You are an interview coach.
Generate one concise, interview-style question for the following round:
{interview_round}

Return only the question text.
"""

    try:
        response = llm.generate([[HumanMessage(content=prompt)]])
        question = response.generations[0][0].text.strip()

        if question:
            return question
    except Exception:
        pass

    return get_default_question()


def evaluate_answer(question, answer):

    if not answer:
        return "Please submit an answer to receive evaluation."

    prompt = f"""
You are an interview evaluator.
The interview question was:
{question}

The candidate answered:
{answer}

Provide a short evaluation with:
- strengths
- areas for improvement
- score out of 10
Return only markdown-formatted text.
"""

    try:
        response = llm.generate([[HumanMessage(content=prompt)]])
        evaluation = response.generations[0][0].text.strip()

        if evaluation:
            return evaluation
    except Exception:
        pass

    return "I couldn't evaluate the answer at this time. Please try again later."


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
        st.session_state.qa_chain = qa_chain

        st.success("✅ Knowledge Base Ready")


# ==========================================
# MOCK INTERVIEW SECTION
# ==========================================

st.header("💬 AI Mock Interview")

# START INTERVIEW BUTTON

if st.button("🎤 Start Interview"):

    if st.session_state.vectorstore is None:

        st.warning("Please build knowledge base first")
        st.stop()

    question = generate_interview_question()

    st.session_state.current_question = question
    st.session_state.interview_started = True

# DISPLAY CURRENT QUESTION

if st.session_state.interview_started:

    st.subheader("🧠 Interview Question")

    st.info(st.session_state.current_question)

    user_answer = st.text_area(
        "Your Answer"
    )

    if st.button("Submit Answer"):

        with st.spinner("Evaluating Answer..."):

            evaluation = evaluate_answer(
                st.session_state.current_question,
                user_answer
            )

            st.markdown("## 📊 Evaluation")

            st.markdown(evaluation)

            # Generate next question

            next_question = generate_interview_question()

            st.session_state.current_question = next_question

            st.markdown("## 🎯 Next Question")

            st.info(next_question)
# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# ==========================================
# QUERY HANDLING
# ==========================================

