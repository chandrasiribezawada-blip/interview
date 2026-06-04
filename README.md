# 🎯 InterviewMentor AI

An AI-powered multi-round mock interview platform built using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **FAISS Vector Database**, **HuggingFace Embeddings**, and **Groq LLMs**.

InterviewMentor AI analyzes uploaded resumes, understands candidate skills and projects, generates personalized interview questions, evaluates answers in real time, maintains conversational memory, and produces a comprehensive interview performance report.

---

# 🚀 Features

## ✅ AI-Powered Mock Interviews
- Conducts realistic industry-style interviews
- Automatically generates questions
- Personalized based on resume content
- Supports adaptive questioning

## ✅ Multi-Round Interview Support

Available Interview Modes:

- Technical Round
- HR Round
- Behavioral Round
- System Design Round
- Aptitude Round

---

## ✅ Resume Understanding Engine

The system automatically analyzes uploaded resumes and identifies:

- Academic Background
- Skills
- Technologies
- Projects
- Certifications
- Experience Level

This enables personalized interview generation.

---

## ✅ Retrieval-Augmented Generation (RAG)

The application uses a complete RAG pipeline:

- PDF Parsing
- Text Chunking
- Embedding Generation
- Vector Storage
- Semantic Retrieval
- Context-Aware Question Generation

This ensures interview questions are grounded in the candidate's resume.

---

## ✅ Conversational Memory

Maintains:

- Previous Questions
- Candidate Answers
- Evaluation History
- Interview Progress

Enables intelligent follow-up questioning.

---

## ✅ AI-Based Candidate Evaluation

Each answer is evaluated based on:

- Technical Accuracy
- Communication Skills
- Clarity
- Confidence
- Depth of Understanding

Provides:

- Score (/10)
- Strengths
- Weaknesses
- Improvement Suggestions

---

## ✅ Dynamic Follow-Up Questions

Unlike traditional interview bots, InterviewMentor AI:

- Remembers previous responses
- Avoids repeated questions
- Generates contextual follow-ups

Creating a realistic interview experience.

---

## ✅ Interview History Tracking

Stores:

- Questions Asked
- Candidate Answers
- AI Evaluations
- Individual Scores

Allows candidates to review their complete interview session.

---

## ✅ Final Interview Report

At the end of the interview:

- Average Score
- Performance Analysis
- Strengths
- Weaknesses
- Areas for Improvement

are automatically generated.

---

## ✅ Persistent Vector Database

Uses FAISS persistence:

- Build once
- Reuse embeddings
- Faster startup
- Reduced processing time

---

## ✅ Duplicate Document Detection

Prevents:

- Duplicate resume uploads
- Duplicate vector indexing
- Redundant embeddings

---

## ✅ Corrupted PDF Handling

Gracefully handles:

- Invalid PDFs
- Empty PDFs
- Unsupported documents

without crashing the application.

---

# 🏗️ Technology Stack

| Technology | Purpose |
|------------|----------|
| Streamlit | Frontend UI |
| Python | Backend Logic |
| LangChain | RAG Framework |
| FAISS | Vector Database |
| HuggingFace Embeddings | Semantic Embeddings |
| Groq Llama 3.3 70B | Large Language Model |
| PyPDFLoader | PDF Processing |
| RecursiveCharacterTextSplitter | Chunking |
| ConversationBufferMemory | Conversation Memory |
| dotenv | Environment Management |

---

# 🧠 AI & RAG Architecture

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

Used for:

- Semantic Search
- Similarity Matching
- Resume Understanding

---

## Chunking Strategy

```text
RecursiveCharacterTextSplitter
```

Parameters:

```python
chunk_size = 1000
chunk_overlap = 200
```

Benefits:

- Preserves context
- Reduces information loss
- Improves retrieval quality

---

## Vector Database

```text
FAISS (Facebook AI Similarity Search)
```

Used for:

- Fast similarity search
- Resume retrieval
- Context retrieval

---

## Retrieval Strategy

```text
MMR (Max Marginal Relevance)
```

Configuration:

```python
search_type="mmr"

k=4
fetch_k=20
lambda_mult=0.7
```

Benefits:

- Reduces duplicate chunks
- Improves diversity
- Better context retrieval

---

# 📊 System Architecture

```text
+----------------------------------------------------+
|                 User Upload Resume                 |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|                 PDF Processing Layer               |
|                  (PyPDFLoader)                     |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|                 Text Chunking Layer                |
|      RecursiveCharacterTextSplitter                |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|              HuggingFace Embeddings                |
|          all-MiniLM-L6-v2 Model                    |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|                  FAISS Vector DB                   |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|              MMR Semantic Retrieval                |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|               Resume Understanding                 |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|          Groq Llama-3.3-70B Question Engine        |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|             Candidate Answer Evaluation            |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|              Conversation Memory Layer             |
+----------------------------------------------------+
                        |
                        v
+----------------------------------------------------+
|             Final Interview Report                 |
+----------------------------------------------------+
```

---

# 🔄 Workflow Diagram

```text
+---------------------+
| Upload Resume PDFs  |
+----------+----------+
           |
           v
+---------------------+
| Process Documents   |
+----------+----------+
           |
           v
+---------------------+
| Create Chunks       |
+----------+----------+
           |
           v
+---------------------+
| Generate Embeddings |
+----------+----------+
           |
           v
+---------------------+
| Store in FAISS DB   |
+----------+----------+
           |
           v
+---------------------+
| Start Interview     |
+----------+----------+
           |
           v
+---------------------+
| Generate Question   |
+----------+----------+
           |
           v
+---------------------+
| Candidate Answers   |
+----------+----------+
           |
           v
+---------------------+
| AI Evaluation       |
+----------+----------+
           |
           v
+---------------------+
| Follow-up Question  |
+----------+----------+
           |
           v
+---------------------+
| Final Report        |
+---------------------+
```

---

# 👤 Use Case Diagram

```text
                 +------------------+
                 |    Candidate     |
                 +---------+--------+
                           |
      +--------------------+------------------+
      |                    |                  |
      v                    v                  v

+-------------+    +---------------+   +---------------+
| Upload PDF  |    | Start         |   | View Final    |
| Resume      |    | Interview     |   | Report        |
+-------------+    +---------------+   +---------------+

                           |
                           v

                +----------------------+
                | InterviewMentor AI   |
                +----------------------+

                           |
          +----------------+----------------+
          |                |                |
          v                v                v

+----------------+ +----------------+ +----------------+
| Generate       | | Evaluate       | | Generate       |
| Questions      | | Answers        | | Feedback       |
+----------------+ +----------------+ +----------------+
```

---

# 🔁 Sequence Diagram

```text
Candidate        Streamlit       FAISS       Groq LLM

    |                |             |            |
    | Upload PDF     |             |            |
    |--------------->|             |            |
    |                | Store Docs  |            |
    |                |-----------> |            |
    |                |             |            |
    | Start Interview|             |            |
    |--------------->|             |            |
    |                | Retrieve    |            |
    |                |-----------> |            |
    |                |             |            |
    |                | Context     |            |
    |                |<----------- |            |
    |                |             |            |
    |                | Generate Qn |----------> |
    |                |             |            |
    |                | Question    |<---------- |
    |<---------------|             |            |
    |                |             |            |
    | Answer         |             |            |
    |--------------->|             |            |
    |                | Evaluate    |----------> |
    |                |             |            |
    |                | Feedback    |<---------- |
    |<---------------|             |            |
```

---

# 🚀 Deployment Diagram

```text
+----------------------+
|      End User        |
+----------+-----------+
           |
           v
+----------------------+
|      Streamlit UI    |
+----------+-----------+
           |
           v
+----------------------+
| InterviewMentor AI   |
|   Application Layer  |
+----------+-----------+
           |
           |
   +-------+-------+
   |               |
   v               v

+-----------+   +-----------+
| FAISS DB  |   | Groq LLM  |
+-----------+   +-----------+

           |
           v

+----------------------+
| Resume Knowledge Base|
+----------------------+
```

---

# 📂 Project Structure

```text
InterviewMentorAI/
│
├── app.py
├── .env
├── requirements.txt
├── faiss_index/
├── README.md
│
└── uploads/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/InterviewMentorAI.git

cd InterviewMentorAI
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 🎯 Future Enhancements

- Voice-Based Interviews
- Real-Time Speech Analysis
- Facial Emotion Detection
- Interview Analytics Dashboard
- Cloud Deployment
- Multi-User Authentication
- Interview Recording
- PDF Report Export

---

# 👩‍💻 Author

**Varshini Bezawada**

B.Tech Information Technology

AI • RAG • LLM Applications • Generative AI

---

# 📜 License

This project is licensed under the MIT License.
