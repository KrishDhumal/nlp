# Auditor AI — Legal Document Summarizer & Auditor

An AI-powered legal-tech platform that analyzes PDFs for risks, generates structured audit reports, enables multi-document comparison, runs compliance checks against global regulations, and lets you **chat with your documents** using Retrieval-Augmented Generation (RAG).

---

## ✨ Features

### Phase 1 — Authentication
- Secure signup/login with JWT tokens + bcrypt password hashing
- Protected routes with persistent sessions (localStorage)

### Phase 2 — Document Processing & Summarization
- Upload PDF documents for automated legal analysis
- **One shared processing pipeline**: extraction, cleaning, structure detection and
  chunking happen exactly once, and the resulting canonical document feeds
  summarization, RAG, clause detection and entity extraction alike
- Page-aware extraction (pdfplumber, PyMuPDF fallback) with page boundaries preserved
  for citation; scanned PDFs are reported as needing OCR, never summarized as empty
- **Token-aware, structure-aware chunking** — sections, then paragraphs, then
  sentences, bounded by a configurable token budget rather than character counts
- **Local-first summarization**: a Hugging Face LED model runs in-process, so
  summarization needs no API key and has no quota. Gemini is an optional provider
- Hierarchical map/reduce summarization: chunk → section → executive summary,
  with depth that adapts to document length
- Structured output (parties, dates, financial terms, obligations) produced by
  deterministic extraction, so every value is a literal substring of the contract
  with an exact page reference
- Background processing with live status and progress; the upload returns immediately
- Content-hash deduplication, scoped per user
- Vector embeddings in Pinecone (384-dim, `all-MiniLM-L6-v2`) with a local fallback
- Optional Gemini clause/risk analysis layered on top
- Interactive dashboard with stat cards and recent audits
- Document library with search and filtering

#### Summarization architecture

```
PDF → validate → extract (page-aware) → clean → detect structure → token-aware chunks
                                                                          │
                                        ┌─────────────────┬───────────────┤
                                        ▼                 ▼               ▼
                                  SUMMARIZATION      RAG INDEX     CLAUSE / ENTITY
                                  (local LED)        (Pinecone)      ANALYSIS
                                        │                 │               │
                                        └────────── MongoDB `audits` ─────┘
```

Summarization runs behind a provider interface:

```
SummarizationProvider
    ├── LocalTransformerSummarizer   (default — no API key)
    └── GeminiSummarizer             (optional)
```

Select with `SUMMARY_PROVIDER=local|gemini`. The model is configured centrally via
`SUMMARIZATION_MODEL` and loaded once per process, never per chunk.

> **Model note:** `SUMMARIZATION_MODEL` must be fine-tuned for summarization *of
> documents like these* — a stricter requirement than "legal" or "long context".
> Two checkpoints that look right and are not:
>
> | Checkpoint | Problem |
> |---|---|
> | `allenai/led-base-16384` | Only pretrained, not fine-tuned. Echoes its input. |
> | `nsi319/legal-led-base-16384` | Fine-tuned on SEC litigation releases. Emits fluent prose about court judgments and insider trading for *any* contract — fabrication, not summarization. |
>
> The default is therefore `Falconsai/text_summarization`, a small (60M) summarization
> fine-tune that stays close to its source. It is largely extractive and modest in
> quality, but faithful — which is the property that matters for a contract. Set
> `SUMMARIZATION_MODEL` to substitute a larger model, and `SUMMARIZATION_FALLBACK_MODELS`
> for what to try if it cannot be loaded.

#### Output is gated before it is shown

Every model output — at chunk, group, section and executive level — must pass three
checks or it is replaced by an extract of the source, which cannot invent anything:

| Check | Catches |
|---|---|
| **Grounding** (≥60% of content words appear in the source) | A checkpoint fine-tuned on the wrong domain, writing fluent text about something else |
| **Invented values** | A monetary amount, percentage, period or date that is not in the source |
| **Mis-paired values** | A figure attached to the wrong words — "Ten Thousand Dollars ($240,000 USD)" — where every value exists in the source but the term has been altered |

When the model cannot summarize most sections, the result is flagged `degraded` with a
reason, and the UI shows a banner. A silent fall back to verbatim text looks identical
to a broken summarizer, so it is never silent.

> **Deployment note:** each worker process that summarizes holds its own copy of
> the model (~1 GB resident for LED-base in fp32). For more than two Uvicorn
> workers, run summarization in a dedicated single-worker service.

### Phase 3 — RAG Chat
- **Chat with Document**: Ask natural-language questions about any audited PDF
- Retrieval-Augmented Generation: Answers grounded in actual document content
- Page-number citations for every response
- Strict anti-hallucination system prompt
- Rate limiting + Redis caching for performance
- Slide-out chat drawer with typing indicators, suggested questions, and source badges

### Phase 4 — Multi-Document Comparison
- **Compare two contracts** side-by-side (e.g., Old vs. New version)
- Clause-by-clause analysis with match/modified/missing status
- **Risk Shift Score** (-100 to +100) showing if the new version is safer
- Missing protections highlighted per document

### Phase 5 — Agentic Compliance Checker
- Proactive compliance auditing against **GDPR**, **CCPA**, and **2026 EU AI Act**
- Specific redline suggestions for each violation
- Prioritized action items for remediation
- Overall compliance score (0-100)

### Phase 6 — Production Hardening
- **PDF/DOCX Export**: Download branded audit reports with cover page, risk table, chat history
- **Redis Caching**: Optional Upstash/Redis chat response caching with graceful fallback
- **Docker**: Full Dockerfile + docker-compose for backend, frontend, and MongoDB
- **Render.com**: Deployment blueprint for one-click cloud hosting
- **Notification System**: Real-time audit event notifications in the header
- **Gemini Model Fallback**: Auto-retry with backup models on API overload (503/429)

---

## 🏗 Architecture

```
Frontend (React + Vite)  ──→  Backend (FastAPI)  ──→  MongoDB (Metadata)
        │                          │                    
        │                          ├──→  Pinecone (Vectors, 384-dim)
        │                          │
        │                          ├──→  Gemini 2.0 Flash (AI)
        │                          │         ↑
        │                          │    HuggingFace Embeddings
        │                          │    (all-MiniLM-L6-v2)
        │                          │
        │                          └──→  Redis/Upstash (Cache, optional)
```

---

## 🚀 Quick Start

> **Full detailed guide**: See [`SETUP.md`](./SETUP.md)

```bash
# 1. Clone & configure
git clone https://github.com/Ani1801/Legal_Document_Summarizer.git
cd Legal_Document_Summarizer
copy .env.example .env     # Fill in your API keys

# 2. Backend
cd backend
python -m venv venv
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/auth/signup` | ❌ | Create new account |
| POST | `/api/auth/login` | ❌ | Login & get JWT |
| GET | `/api/dashboard/stats` | ✅ | Dashboard statistics |
| GET | `/api/audits/recent` | ✅ | Recent 5 audits |
| POST | `/api/audits/upload` | ✅ | Upload PDF & run audit |
| GET | `/api/audits/file/{id}` | ✅ | Serve PDF for preview |
| GET | `/api/library` | ✅ | All user documents |
| GET | `/api/notifications` | ✅ | Audit event notifications |
| POST | `/api/chat` | ✅ | RAG chat with document |
| POST | `/api/compare` | ✅ | Compare two documents |
| POST | `/api/compliance/check` | ✅ | Run compliance check |
| GET | `/api/export/{id}?format=pdf` | ✅ | Export report as PDF |
| GET | `/api/export/{id}?format=docx` | ✅ | Export report as DOCX |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Tailwind CSS 3.4, Framer Motion |
| Backend | FastAPI, Python 3.11+, Uvicorn |
| Auth | PyJWT, bcrypt |
| Database | MongoDB (Motor async driver) |
| Vector DB | Pinecone (Starter, 384-dim, cosine) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | Google Gemini 2.0 Flash (free tier) |
| PDF Parser | PyMuPDF |
| Reports | ReportLab (PDF), python-docx (DOCX) |
| Caching | Redis / Upstash (optional) |
| Orchestration | LangChain |
| Deployment | Docker, Render.com |

---

## 📄 License

This project is for educational purposes.

## 🧪 Tests

Two suites cover the document pipeline. Both run with **no MongoDB, no Pinecone,
no API key and no model download** — the database and the summarization model are
replaced with in-memory stand-ins, while extraction, cleaning, structure detection,
chunking, embedding, the routes and the response schema all run as real code.

```bash
cd backend
PYTHONPATH=. ./venv/bin/python tests/test_audit_pipeline.py          # 92 checks
PYTHONPATH=. ./venv/bin/python tests/test_summarization_pipeline.py  # 148 checks
```

