# agentic-rag

An agentic Retrieval-Augmented Generation (RAG) system that extends a baseline RAG pipeline with tool use, query routing, multi-step reasoning, and conversational memory. Built to demonstrate the progression from passive retrieval to an active, decision-making AI agent.

---

## Overview

Where a standard RAG pipeline retrieves context and generates a single response, an agentic RAG system reasons over the query, decides which tools or retrieval paths to use, reflects on intermediate results, and iterates until it reaches a satisfactory answer. This repo showcases that full reasoning loop.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Multiple (OpenAI GPT-4o, Anthropic Claude, etc.) |
| Embeddings | OpenAI / HuggingFace |
| Vector Store | Pinecone (cloud managed) |
| Memory | In-memory (conversation history) |
| Framework | Python |
| Demo UI | Streamlit |

---

## Repo Structure

```
agentic-rag/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   └── chunker.py
│   ├── embedding/
│   │   └── embedder.py
│   ├── vectorstore/
│   │   └── store.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── generation/
│   │   └── generator.py
│   ├── agents/
│   │   ├── agent.py
│   │   ├── tools.py
│   │   └── router.py
│   ├── memory/
│   │   └── memory.py
│   └── pipeline.py
├── evaluation/
│   └── eval.py
├── notebooks/
│   └── exploration.ipynb
├── ui/
│   └── app.py
├── tests/
│   └── test_pipeline.py
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Getting Started

**Prerequisites**
- Python 3.10+
- OpenAI or Anthropic API key
- Pinecone API key

**Installation**

```bash
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

**Run the pipeline**

```bash
python src/pipeline.py
```

**Launch the demo UI**

```bash
streamlit run ui/app.py
```

---

## Key Design Decisions

**Agent reasoning loop** — the agent evaluates the query, selects the appropriate tool or retrieval path via the router, reflects on intermediate results, and re-retrieves or re-reasons if the answer is insufficient. This loop is explicit and traceable in the code.

**Query routing** — rather than always hitting the vector store, the router classifies the query and directs it to the most appropriate tool — retrieval, web search, calculator, or direct LLM response.

**Conversational memory** — the agent maintains a running conversation history passed as context on each LLM call. This allows multi-turn reasoning without losing prior context.

**Multiple LLMs** — same as the baseline pipeline, the generation layer supports swapping between providers to compare agent behavior across models.

**Streamlit UI** — chosen over Gradio to visualize the agent's reasoning steps, tool calls, and intermediate outputs as a dashboard alongside the final response.

---

## Evaluation

The `evaluation/` module measures retrieval precision and recall, answer faithfulness and relevance, agent tool selection accuracy, and reasoning step count per query.

---

## Production Considerations

This project is intentionally scoped for demonstration. In a production system:

- **Vector store** — Pinecone would be configured with namespaces and metadata filtering for multi-tenant support and more precise retrieval.
- **Memory** — in-memory conversation history would be replaced by Redis for persistent, low-latency session storage across multiple users and requests.
- **API layer** — the agent pipeline would be exposed via a FastAPI service with async support to handle the latency of multi-step reasoning without blocking.
- **Frontend** — the Streamlit demo would be replaced by a React or Next.js frontend with streaming support for displaying agent reasoning steps in real time.
- **Observability** — LangSmith or Arize would be added for tracing each reasoning step, tool call, and retrieval result in production.

---

## Related Project

This repo builds directly on [rag-pipeline](https://github.com/tohio/rag-pipeline), which covers the baseline RAG implementation this system extends.
