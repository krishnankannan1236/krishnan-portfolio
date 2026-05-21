# Krishnan Kannan — Portfolio + Resume RAG Chatbot

A Streamlit personal portfolio with an interactive chatbot that answers questions grounded in the resume using a Retrieval-Augmented Generation (RAG) pipeline.

## Architecture

```
data/resume.txt
      │
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ section-aware│ -> │  MiniLM      │ -> │  FAISS       │
│   chunker    │    │  embeddings  │    │  index       │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
                            user question ──────┤
                                                ▼
                                        top-k retrieval
                                                │
                                                ▼
                              ┌──────────────────────────────┐
                              │  Claude / GPT — answers       │
                              │  grounded in retrieved chunks │
                              └──────────────────────────────┘
```

- **Chunking** is section-aware: it respects `##` / `###` headings in the resume so each chunk carries a meaningful label like *"Experience — Merck Group"*.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — runs locally, free, 384-dim.
- **Vector store**: FAISS `IndexFlatIP` on normalised vectors = cosine similarity.
- **Generation**: Anthropic Claude Haiku 4.5 by default; OpenAI also supported.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# choose one:
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

streamlit run app.py
```

The app also accepts the API key directly in the UI (session-only, not persisted).

## Deploy to Streamlit Community Cloud — free

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io and connect the repo.
3. Set `app.py` as the entry point.
4. In **Settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Deploy. First load takes ~1 min while sentence-transformers downloads the model.

## File map

```
portfolio/
├── app.py              # Streamlit UI: hero, tabs, chat
├── rag.py              # chunk + embed + retrieve + generate
├── requirements.txt
├── data/
│   └── resume.txt      # source of truth — edit to update content
└── .streamlit/
    └── config.toml     # theme
```

## Updating the resume

Edit `data/resume.txt`. Restart the app (or click *Rerun*) — the cached index rebuilds on next load.

## Why these choices

- **MiniLM over OpenAI embeddings**: free, runs anywhere, fully adequate for ~30 chunks. Good for cost and privacy.
- **FAISS over Chroma/Pinecone**: no server, no vendor lock-in, ~30 chunks doesn't justify external infra.
- **Section-aware chunks over fixed-size**: a resume's structure is its meaning. Chunking on `###` keeps every chunk self-describing.
- **Claude Haiku for generation**: cheap, fast, strong at grounded summarisation.
