# Persian Tax Assistant: RAG System & FastAPI Service

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions about Iranian tax regulations in Persian (Farsi), backed by a FastAPI service, a Streamlit UI, and a full test suite using PyTest.

---

## Overview

This project implements an end-to-end RAG pipeline that:
- Processes Persian tax documentation into semantically searchable chunks
- Uses multilingual sentence embeddings to retrieve relevant context
- Routes queries through a FastAPI backend to free LLM providers via OpenRouter
- Renders a bilingual response (Persian primary, English translation below)
- Exposes a clean Streamlit chat interface for end users
- Ships with pytest integration tests for the API layer

### Dataset Creation

No ready-made Persian tax dataset existed, so a **custom web crawler** was built to scrape publicaly available data from [tax.gov.ir](https://tax.gov.ir) — the official Iranian Tax Authority website. The crawler navigates documentation pages, extracts and cleans Persian text, and produces `inta_texts_cleaned.txt`. This demonstrates **end-to-end data engineering** from raw government HTML to a production knowledge base.

---
### Component Breakdown

**`Data collection`** — Custom crawler scrapes and cleans tax.gov.ir into `inta_texts_cleaned.txt`.

**`main.py`** — The original standalone prototype that runs the full RAG pipeline directly in the terminal with no API or UI. Handles ingestion (chunking + embedding + ChromaDB storage) and runs an interactive Persian Q&A loop in the console. Useful for testing the pipeline locally without spinning up any services.

**`FastAPI Backend` (api.py)**
The core service is optimized for performance by loading the embedding models into memory once at startup. It features an LLM Fallback Chain to ensure reliability; if the primary model (e.g., Qwen 3 or Llama 3.3) is unavailable via OpenRouter, the system automatically cycles through a priority list of 10+ models to ensure a successful response.

**`Streamlit UI` (app.py)**
A lightweight, reactive frontend that interacts with the FastAPI layer. It provides a clean chat interface that renders bilingual answers (Persian primary with an English translation).

**`Testing & Reliability (test.py)`**
Using pytest and FastAPI's TestClient, the project includes integration tests that verify:

API endpoint health and response latency.

Pydantic schema validation for input/output.

Edge cases like empty queries or malformed requests.

---

## LLM Fallback Chain

Models are tried in order until one responds successfully:

1. Qwen 3 14B
2. OpenRouter Auto-select
3. Meta Llama 3.3 70B
4. DeepSeek R1
5. DeepSeek V3
6. NVIDIA Nemotron 120B
7. Qwen 3 32B
8. Google Gemma 4 31B / Gemma 3 27B / Gemma 3 12B / Gemma 3 4B
9. Mistral Small 3.1 24B
10. Meta Llama 3.1 8B

---


## Project Structure

```text
├── main.py                  # Standalone prototype: ingestion + terminal Q&A loop
├── api.py                   # FastAPI backend: /ask endpoint
├── app.py                   # Streamlit UI
├── test.py                  # pytest API tests
├── inta_texts_cleaned.txt   # Scraped and cleaned tax documentation
├── chroma_db/               # Persistent vector database
├── .env                     # OPENROUTER_API_KEY (not committed)
├── requirements.txt
└── README.md
```
---

## Quick Start
1. **Setup:** `pip install -r requirements.txt`
2. **Environment:** Add `OPENROUTER_API_KEY` to `.env`.
3. **Launch:** `uvicorn api:app` for the backend and `streamlit run app.py` for the UI.
4. **Test:** Run `pytest` to verify the RAG pipeline.


---

## Skills Demonstrated

- Data Engineering: End-to-end pipeline from raw web scraping to structured vector storage.

- RAG Architecture: Implementing semantic retrieval, document chunking, and context injection.

- Defensive Programming: Building a multi-model fallback system for third-party API reliability.

- Full-Stack AI: Decoupling logic into a RESTful API and a modern web interface.

- Software Rigor: Applying automated testing to AI workflows to ensure production stability.
