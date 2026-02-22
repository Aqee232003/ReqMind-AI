🚀 ReqMind AI
AI-Powered Business Requirements Document Generator

RAG | FastAPI | FAISS | Gemini | Cloud Deployment

🔍 Overview

ReqMind AI is a production-ready Retrieval-Augmented Generation (RAG) system that automatically transforms unstructured communication (meeting transcripts, emails, Slack chats) into structured, traceable, enterprise-grade Business Requirements Documents (BRDs).

It demonstrates applied skills in:

Large Language Models (LLMs)

RAG architecture design

Vector search (FAISS)

Prompt engineering

Backend API development (FastAPI)

Cloud deployment (Google Cloud Run)

System pipeline design

Confidence-based human-in-the-loop validation

🎯 Problem Statement

Requirement documentation in real-world projects is:

Manual and time-consuming

Prone to ambiguity and misinterpretation

Difficult to trace back to original discussions

Lacking structured conflict detection

ReqMind AI automates this process while preserving traceability and interpretability.

🏗️ System Architecture

The system is built as a 9-stage intelligent pipeline:

Input Ingestion – Supports transcripts, email threads, Slack chats, AMI dataset

Rule-Based Preprocessing – Speaker segmentation + filler removal

RAG Chunking – Semantic chunk creation with overlap

Embedding Generation – SentenceTransformers (MiniLM)

Vector Indexing – FAISS for similarity search

LLM Extraction – Gemini extracts structured requirements

Conflict Detection – LLM-assisted contradiction detection

BRD Template Generation – Structured enterprise-ready format

Executive Summary Generation – Context-aware summarization

🧠 Technical Stack
Component	Technology
Backend API	FastAPI
LLM	Gemini (gemini-2.0-flash)
Embeddings	all-MiniLM-L6-v2
Vector Database	FAISS
Deployment	Google Cloud Run
Containerization	Docker
📊 Core Capabilities

✔ Functional & Non-Functional Requirement Extraction
✔ Stakeholder Identification & Sentiment Mapping
✔ Timeline & Milestone Detection
✔ Conflict & Risk Detection
✔ Confidence Scoring per Requirement
✔ Human-in-the-Loop (HITL) Flagging
✔ Traceability Matrix Generation
✔ Iterative AI-Based BRD Editing
✔ Multi-source Input Normalization

Each requirement includes:

1. Unique ID

2. Source speaker

3. Timestamp

4. Confidence score

5. Source traceability

☁️ Deployment-Ready Architecture

The application is fully containerized and deployed on Google Cloud Run, demonstrating:

Environment-based configuration

Stateless API architecture

Production-safe CORS handling

Graceful LLM fallback handling (mock/live mode)

Rate limit resilience

🛠️ Running Locally
uvicorn main:app --reload --port 8000
🐳 Docker
docker build -t reqmind-ai .
docker run -p 8000:8000 reqmind-ai
📈 Engineering Highlights

Designed and implemented full RAG pipeline from scratch

Integrated FAISS vector search with semantic retrieval

Built structured JSON schema enforcement for LLM outputs

Implemented fallback handling for LLM rate limits

Designed confidence-based ambiguity flagging system

Developed enterprise-grade BRD templating engine

Ensured separation of rule-based vs LLM-driven logic


📌 Why This Project Matters

1. This project demonstrates:

2. Applied AI system design

3. Real-world NLP engineering

4. LLM production integration

5. Backend architecture skills

6. Cloud deployment knowledge

7. Strong understanding of requirement engineering workflows

8. It bridges AI research concepts with practical enterprise application.

👩‍💻 Team

CodeBlooded
HackFest 2.0 Submission
