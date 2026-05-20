import os
import time
import streamlit as st

try:
    from groq import APIConnectionError
except ImportError:
    APIConnectionError = Exception

from .config.settings import (
    APP_TITLE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    GROQ_API_KEY,
    LLM_MODEL_NAME,
)
from .ingestion.chunker import split_documents
from .ingestion.embeddings import create_embeddings
from .ingestion.pdf_loader import load_pdf_documents
from .interview.evaluator import analyze_resume, evaluate_answer
from .interview.feedback import summarize_interview_history
from .interview.question_generator import generate_interview_question
from .llm.llm_factory import create_llm
from .memory.conversation_memory import create_conversation_memory
from .pipeline.rag_pipeline import build_qa_chain
from .utils.helpers import unique_documents
from .vectorstores.faiss_store import create_faiss_store, load_faiss_store


@st.cache_resource
def load_embedding_model():
    return create_embeddings(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_llm():
    return create_llm(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.3,
    )


def process_documents(files, embeddings, chunk_size, chunk_overlap):
    if chunk_overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size")
        st.stop()

    all_docs = load_pdf_documents(files)
    split_docs = split_documents(all_docs, chunk_size, chunk_overlap)
    split_docs = unique_documents(split_docs)

    vectorstore = create_faiss_store(split_docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

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

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False

    if "memory" not in st.session_state:
        st.session_state.memory = create_conversation_memory()

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

    st.sidebar.header("⚙ Interview Configuration")

    interview_round = st.sidebar.selectbox(
        "Select Interview Round",
        [
            "HR Round",
            "Technical Round",
            "System Design Round",
            "Behavioral Round",
            "Aptitude Round",
        ],
    )

    top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 4)
    chunk_size = st.sidebar.slider("Chunk Size", 200, 2000, 1000)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 200)

    st.sidebar.slider(
        "Number of Questions",
        5,
        10,
        25,
        key="max_questions",
    )

    uploaded_files = st.file_uploader(
        "📄 Upload Resume / Notes / PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    embeddings = load_embedding_model()

    if not GROQ_API_KEY:
        st.sidebar.error(
            "GROQ_API_KEY is not configured. "
            "Set GROQ_API_KEY in InterviewMentorAI/.env or the root .env."
        )
        st.stop()

    llm = load_llm()

    if os.path.exists(FAISS_INDEX_PATH):
        if st.session_state.vectorstore is None:
            loaded = load_faiss_store(embeddings, FAISS_INDEX_PATH)
            if loaded is not None:
                st.session_state.vectorstore = loaded

    if st.button("🚀 Build Interview Knowledge Base"):
        if not uploaded_files:
            st.warning("Please upload PDF files")
            st.stop()

        with st.spinner("Processing Documents..."):
            vectorstore = process_documents(
                uploaded_files,
                embeddings,
                chunk_size,
                chunk_overlap,
            )

            qa_chain = build_qa_chain(
                llm,
                vectorstore,
                interview_round,
                top_k,
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = qa_chain

            try:
                st.session_state.resume_summary = analyze_resume(
                    llm, vectorstore
                )
            except APIConnectionError as error:
                st.error(
                    "Unable to connect to GROQ API. "
                    "Check your GROQ_API_KEY, internet connection, and proxy settings."
                )
                st.stop()
            except Exception as error:
                st.error(f"Error analyzing resume: {error}")
                st.stop()

            st.success("✅ Knowledge Base Ready")
            st.subheader("📄 Resume Understanding")
            st.write(st.session_state.resume_summary)

    st.header("💬 AI Mock Interview")

    if st.button("🎤 Start Interview"):
        if st.session_state.vectorstore is None:
            st.warning("Please build knowledge base first")
            st.stop()

        st.session_state.question_number = 1
        st.session_state.total_score = 0
        st.session_state.interview_history = []
        st.session_state.interview_completed = False

        question = generate_interview_question(
            llm=llm,
            vectorstore=st.session_state.vectorstore,
            interview_round=interview_round,
            resume_summary=st.session_state.resume_summary,
            memory=st.session_state.memory,
            top_k=top_k,
        )

        st.session_state.current_question = question
        st.session_state.interview_started = True

    if st.session_state.interview_history:
        st.subheader("📜 Interview History")
        for idx, item in enumerate(
            st.session_state.interview_history,
            start=1,
        ):
            with st.expander(f"Question {idx}"):
                st.markdown(f"### 🧠 Question\n{item['question']}")
                st.markdown(f"### 👤 Your Answer\n{item['answer']}")
                st.markdown(f"### 📊 Evaluation\n{item['evaluation']}")
                st.success(f"Score: {item['score']}/10")

    if (
        st.session_state.interview_started
        and not st.session_state.interview_completed
    ):
        st.subheader(
            f"🧠 Question {st.session_state.question_number}/"
            f"{st.session_state.max_questions}"
        )
        st.info(st.session_state.current_question)

        user_answer = st.text_area(
            "Your Answer",
            key=f"answer_box_{st.session_state.question_number}",
        )

        if st.button("Submit Answer"):
            if not user_answer.strip():
                st.warning("Please enter your answer")
                st.stop()

            with st.spinner("Evaluating Answer..."):
                start_time = time.time()
                evaluation, score = evaluate_answer(
                    llm,
                    st.session_state.current_question,
                    user_answer,
                    st.session_state.memory,
                    st.session_state.interview_history,
                )
                st.session_state.total_score += score
                end_time = time.time()
                total_latency = round(end_time - start_time, 2)

                st.markdown("## 📊 Evaluation")
                st.markdown(evaluation)
                st.info(
                    f"⏱ Response Evaluation Time: {total_latency} seconds"
                )

                st.session_state.question_number += 1
                if st.session_state.question_number > st.session_state.max_questions:
                    st.session_state.interview_completed = True
                else:
                    next_question = generate_interview_question(
                        llm=llm,
                        vectorstore=st.session_state.vectorstore,
                        interview_round=interview_round,
                        resume_summary=st.session_state.resume_summary,
                        memory=st.session_state.memory,
                        top_k=top_k,
                    )
                    st.session_state.current_question = next_question

                st.rerun()

    if st.session_state.interview_completed:
        st.divider()
        st.header("🏁 Interview Completed")
        average_score = round(
            st.session_state.total_score / st.session_state.max_questions,
            2,
        )
        st.success(f"🎯 Final Interview Score: {average_score}/10")

        if average_score >= 8:
            st.balloons()
            st.success("Excellent performance!")
        elif average_score >= 6:
            st.info("Good performance with room for improvement.")
        else:
            st.warning("Need more preparation.")

        strengths, weaknesses = summarize_interview_history(
            st.session_state.interview_history
        )

        st.subheader("📌 Overall Interview Summary")
        st.markdown("### ✅ Strengths")
        st.write(strengths)
        st.markdown("### ⚠ Areas to Improve")
        st.write(weaknesses)


if __name__ == "__main__":
    main()
