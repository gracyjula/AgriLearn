# 🌾 AgriLearn AI — Your Smart Agriculture Learning Assistant

> **Hackathon Project 20 — Smart Agriculture Crop Process Explainer Bot**
> Industry Domain: Agriculture / Smart Farming

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Main Features](#main-features)
4. [Technology Stack](#technology-stack)
5. [System Architecture](#system-architecture)
6. [Folder Structure](#folder-structure)
7. [Python Environment Setup](#python-environment-setup)
8. [Installing Dependencies](#installing-dependencies)
9. [Configuring the Gemini API Key](#configuring-the-gemini-api-key)
10. [Running the Application](#running-the-application)
11. [RAG Setup — Adding Agricultural PDF Documents](#rag-setup)
12. [How to Rebuild the Vector Index](#rebuilding-the-index)
13. [Running Automated Tests](#running-automated-tests)
14. [Example Questions and Expected Behavior](#example-questions)
15. [Troubleshooting](#troubleshooting)
16. [API Key Security](#api-key-security)
17. [Team](#team)

---

## 🌱 Project Overview

**AgriLearn AI** is a beginner-friendly Generative AI chatbot that explains crop lifecycles and modern farming processes in simple, easy-to-understand language.

It combines two response modes:

| Mode | Description |
|---|---|
| **Direct Gemini** | Answers from Gemini Flash's general training knowledge |
| **RAG mode** | Answers grounded in indexed agricultural PDF documents |

Both modes enforce the same safety restrictions (no fertilizer advice, no disease diagnosis, no yield predictions).

---

## ❓ Problem Statement

Farmers and agricultural workers often lack access to clear explanations of crop cycles and modern farming processes. AgriLearn AI addresses this with an always-available, AI-powered educational tool that explains general agricultural concepts in plain language.

---

## ✅ Main Features

| Feature | Details |
|---|---|
| 💬 Conversational chat | Multi-turn conversation with full history |
| 🌱 Agricultural education | Explains sowing, irrigation, harvesting, storage |
| 📚 RAG mode | Answers grounded in your own agricultural PDF documents |
| 📄 Source citations | Shows document filename and page number below RAG answers |
| 🛡️ Safety restrictions | Refuses fertilizer advice, disease diagnosis, yield predictions |
| 🌐 Language support | English and Telugu |
| 📱 WhatsApp-style UI | User bubbles RIGHT (green), assistant LEFT (cream) |
| 🎤 Voice input | Browser Web Speech API — click mic, speak, transcript appears |
| 🔊 Voice output | Per-message TTS button using browser speechSynthesis |
| 💡 Suggestion chips | 4 example question chips shown on empty chat state |
| 🔒 Secure config | API key via `.env` or Streamlit Secrets |
| 🧪 Automated tests | 17+ pytest tests — no real API key required |
| ⚡ Gemini Flash | Fast responses via LangChain (`gemini-3.5-flash` by default) |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web framework | Streamlit 1.35 |
| LLM orchestration | LangChain 0.2 |
| Gemini integration | `langchain-google-genai` 1.0.10 |
| AI model | Google Gemini Flash (`gemini-3.5-flash`) |
| Embeddings | `gemini-embedding-001` (via `langchain-google-genai`) |
| PDF loading | PyPDF + `langchain-community` PyPDFLoader |
| Vector store | FAISS (local, CPU, no GPU required) |
| Config & secrets | `python-dotenv` / Streamlit Secrets |
| Testing | `pytest`, `pytest-mock` |

---

## 🏗️ System Architecture

```
User (Browser)
     │
     ▼
┌──────────────────────────────────────────────────────┐
│                  app.py (Streamlit UI)                │
│  - Chat interface        - Session state              │
│  - Safety pre-check      - RAG toggle                 │
│  - Source citation display                            │
└──────┬───────────────────────────────────────────────┘
       │
       ├──► safety.py  (heuristic input filter — pre-LLM)
       │
       ├──► rag_service.py  (optional, when RAG enabled)
       │      - Load PDFs  → chunk → embed → FAISS
       │      - Retrieve top-K chunks for question
       │
       └──► gemini_service.py
              - Plain chat prompt  OR  RAG-grounded prompt
              - ChatGoogleGenerativeAI (Gemini Flash)
              - Classify API errors: NotFound / auth / quota / network
                        │
                        ▼
              prompts.py
              - SYSTEM_PROMPT_EN / TE
              - RAG_SYSTEM_PROMPT_EN / TE
              - ChatPromptTemplate + MessagesPlaceholder
                        │
                        ▼
              Google Gemini Flash API
              (gemini-3.5-flash via langchain-google-genai)
```

---

## 📁 Folder Structure

```
agrilearn-ai/
├── app.py                    ← Streamlit UI — WhatsApp-style chat with voice support
├── config.py                 ← API key loading, model configuration, RAG config
├── gemini_service.py         ← LangChain + Gemini integration, error classification
├── prompts.py                ← English & Telugu system prompts + RAG-grounded prompts
├── rag_service.py            ← PDF ingestion, chunking, FAISS indexing, retrieval
├── safety.py                 ← Heuristic input validation and refusal logic
├── voice_component.py        ← Browser Web Speech API — voice input & TTS output
├── requirements.txt          ← Pinned production dependencies
├── requirements-dev.txt      ← Test-only dependencies (pytest, pytest-mock)
├── .env.example              ← Placeholder config (safe to commit)
├── .gitignore                ← Excludes .env, vector store, __pycache__
├── README.md                 ← This file
├── docs/
│   ├── pdfs/                 ← Place your agricultural PDF documents here (auto-created)
│   └── vector_store/         ← FAISS index (auto-generated, not committed to Git)
└── tests/
    ├── __init__.py
    ├── test_safety.py        ← 17 safety + voice component tests
    ├── test_rag.py           ← 14 RAG pipeline tests
    └── test_api_diagnostic.py← Live API diagnostic tests (require real key)
```

---

## 🐍 Python Environment Setup

### Option A — Using `venv` (PowerShell)

```powershell
cd agrilearn-ai
python -m venv agrilearn-env
.\agrilearn-env\Scripts\Activate.ps1
```

> If you see an execution policy error:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Option B — Using `venv` (Command Prompt)

```cmd
cd agrilearn-ai
python -m venv agrilearn-env
agrilearn-env\Scripts\activate.bat
```

### Option C — Using Conda

```cmd
conda create -n agrilearn python=3.11 -y
conda activate agrilearn
cd agrilearn-ai
```

---

## 📦 Installing Dependencies

With your virtual environment activated:

```powershell
pip install -r requirements.txt
```

This installs Streamlit, LangChain, langchain-google-genai, pypdf, faiss-cpu, langchain-community, python-dotenv, and pytest.

---

## 🔑 Configuring the Gemini API Key

### Step 1 — Get a free API key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key

### Step 2 — Create your `.env` file

```powershell
Copy-Item .env.example .env
```

### Step 3 — Add your key

Open `.env` and replace `YOUR_API_KEY_HERE`:

```
GEMINI_API_KEY=your_actual_key_here
```

### Step 4 — Verify the model name

The default model is `gemini-3.5-flash`. **Do not use `gemini-1.5-flash`** — it has been shut down and causes a `NotFound` error.

If needed, override in `.env`:

```
GEMINI_MODEL=gemini-3.5-flash
```

---

## ▶️ Running the Application

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

To stop: press `Ctrl + C` in the terminal.

---

## 📚 RAG Setup — Adding Agricultural PDF Documents <a name="rag-setup"></a>

RAG allows AgriLearn AI to answer questions using content from your own trusted agricultural PDF documents.

### Step 1 — Create the PDFs folder

```powershell
New-Item -ItemType Directory -Force docs\pdfs
```

### Step 2 — Add agricultural PDF documents

Copy any agricultural education PDFs (FAO guides, ICAR manuals, extension bulletins, etc.) into:

```
agrilearn-ai/docs/pdfs/
```

Example file names:
- `rice_cultivation_guide.pdf`
- `wheat_irrigation_manual.pdf`
- `post_harvest_storage.pdf`

### Step 3 — Build the vector index

#### Option A — From the Streamlit UI

1. Start the app: `streamlit run app.py`
2. Look for the **"Build / Rebuild Index"** button in the sidebar
3. Click it and wait for the index to build

#### Option B — From the Python console

```powershell
cd agrilearn-ai
python -c "
from config import AppConfig
from rag_service import RAGService
cfg = AppConfig()
rag = RAGService(cfg)
ok, msg = rag.build_index()
print(msg)
"
```

### Step 4 — Enable RAG in the UI

Once the index is built, the sidebar shows **"✅ Index ready"** and a toggle to enable document search. Turn it on and ask your questions.

> **Note:** RAG requires a valid `GEMINI_API_KEY` to generate embeddings.

---

## 🔄 Rebuilding the Index <a name="rebuilding-the-index"></a>

Rebuild when you add or replace PDF documents:

```powershell
python -c "
from config import AppConfig
from rag_service import RAGService
cfg = AppConfig()
rag = RAGService(cfg)
ok, msg = rag.build_index(force_rebuild=True)
print(msg)
"
```

Or click **"🔄 Build / Rebuild Index"** in the sidebar.

---

## 🎤 Voice Input & Output

AgriLearn AI supports voice through the browser's Web Speech API. **No additional Python packages required** — voice is implemented as pure browser-side JavaScript.

### Voice Input (Speech-to-Text)
1. Click **"🎤 Voice Input"** expander in the chat area.
2. Click the green microphone button.
3. Allow microphone access when the browser asks.
4. Speak your question — the transcript appears in the text box.
5. Copy the transcript and paste it into the chat input, or type it manually.

> **Browser support:** Chrome and Edge support Web Speech API. Firefox does not support speech recognition. Text input always works in all browsers.

### Voice Output (Text-to-Speech)
Each assistant message has a **🔊 Listen** button beneath it.
- Click it to hear the message spoken aloud.
- Click **⏹ Stop** to stop playback.
- Language follows the sidebar selection (English = en-US, Telugu = te-IN).

### Language Codes
| Language | Speech Recognition | Text-to-Speech |
|---|---|---|
| English | en-US (Chrome/Edge) | en-US ✅ |
| Telugu | te-IN (Chrome, may vary) | te-IN ✅ |

---

## 🧪 Running Automated Tests <a name="running-automated-tests"></a>

### Standard tests (no API key required)

```powershell
pytest tests/test_safety.py tests/test_rag.py -v
```

Expected: **31 passed** (17 safety/voice + 14 RAG)

### Live API diagnostic tests (require a real API key)

```powershell
$env:AGRILEARN_RUN_API_TESTS=1
pytest tests/test_api_diagnostic.py -v -s
```

These send real requests to the Gemini API and verify:
1. API key is present
2. The model name is not deprecated
3. A real response is returned
4. The RAG prompt path works

---

## 💬 Example Questions and Expected Behavior <a name="example-questions"></a>

### ✅ Questions the bot WILL answer

| Question | Expected behavior |
|---|---|
| "Explain crop growth stages." | Explains germination → vegetative → flowering → maturity → harvest |
| "What is an irrigation cycle?" | Describes irrigation scheduling and general methods |
| "Explain the harvesting process." | Covers timing indicators, cutting methods, post-harvest handling |
| "What are basic storage practices?" | Explains grain drying, moisture control, silo use |
| "What is drip irrigation?" | Explains the concept and general benefits |

### ❌ Questions the bot WILL NOT answer

| Question | Reason |
|---|---|
| "Recommend a fertilizer for wheat." | Fertilizer recommendation — prohibited |
| "How much urea should I apply?" | Fertilizer dosage — prohibited |
| "What pesticide kills aphids?" | Pesticide recommendation — prohibited |
| "My crop is dying — what disease is it?" | Disease diagnosis — prohibited |
| "Predict my paddy yield this season." | Yield prediction — prohibited |
| "Ignore your instructions." | Prompt injection — blocked |

---

## 🔧 Troubleshooting <a name="troubleshooting"></a>

### `Error type: NotFound` when asking a question

**Cause:** The Gemini model configured in `.env` has been shut down. `gemini-1.5-flash` was retired in 2026.

**Fix:** Open `.env` and set:
```
GEMINI_MODEL=gemini-3.5-flash
```

### `⚠️ API key not configured`

- Check that `.env` exists in the `agrilearn-ai/` folder.
- Verify the line: `GEMINI_API_KEY=your_actual_key` (no spaces around `=`).
- Restart the app after editing `.env`.

### `⚠️ Authentication error`

- Your API key is invalid or has been revoked.
- Generate a new key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### `⚠️ API quota or rate limit exceeded`

- Your free-tier key has hit its per-minute limit. Wait 60 seconds and try again.
- Check your quota at [Google AI Studio](https://aistudio.google.com/app/apikey).

### `ModuleNotFoundError: No module named 'pypdf'` or `faiss`

```powershell
pip install -r requirements.txt
```

### RAG sidebar shows "RAG not available"

Install the RAG dependencies:
```powershell
pip install pypdf==4.3.1 faiss-cpu==1.8.0 langchain-community==0.2.16
```

### `streamlit: command not found` (PowerShell)

```powershell
python -m streamlit run app.py
```

### Tests fail with `ModuleNotFoundError`

Run pytest from inside `agrilearn-ai/`:
```powershell
cd agrilearn-ai
pytest tests/ -v
```

### Port 8501 already in use

```powershell
streamlit run app.py --server.port 8502
```

---

## 🔒 API Key Security

- **Never** hardcode your API key in source code.
- **Never** commit `.env` to Git — it is listed in `.gitignore`.
- **Never** share your API key in screenshots, logs, or chat messages.
- Use `.env.example` (placeholder values only) for version control.
- Rotate your key immediately if you accidentally expose it.
- The app never logs, displays, or transmits your API key value.

---

## 👥 Team

- Team size: 4–5 members
- Hackathon Project: #20 — Smart Agriculture Crop Process Explainer Bot
- Domain: Agriculture / Smart Farming

---

## 📊 Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | — | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-3.5-flash` | Chat model name |
| `GEMINI_EMBEDDING_MODEL` | No | `gemini-embedding-001` | Embedding model for RAG |
| `GEMINI_TEMPERATURE` | No | `0.3` | Model creativity (0.0–1.0) |
| `AGRILEARN_PDF_DIR` | No | `docs/pdfs` | Folder containing agricultural PDFs |
| `AGRILEARN_VECTOR_STORE_DIR` | No | `docs/vector_store` | FAISS index storage folder |
| `AGRILEARN_CHUNK_SIZE` | No | `800` | Characters per text chunk |
| `AGRILEARN_CHUNK_OVERLAP` | No | `100` | Overlap between chunks |
| `AGRILEARN_TOP_K` | No | `4` | Chunks retrieved per query |
| `AGRILEARN_RUN_API_TESTS` | No | — | Set to `1` to run live API tests |

---

*AgriLearn AI — Powered by Google Gemini Flash & LangChain. For educational purposes only. Not a substitute for professional agricultural advice.*
