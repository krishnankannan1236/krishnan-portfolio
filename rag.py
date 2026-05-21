"""
rag.py — Resume RAG pipeline.

Pipeline stages:
1. Load   — read resume.txt
2. Chunk  — section-aware splitting (preserves resume structure)
3. Embed  — sentence-transformers (local, no API key)
4. Index  — FAISS cosine similarity
5. Retrieve — top-k chunks for a query
6. Generate — LLM answer grounded in retrieved chunks (Anthropic or OpenAI)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# Lazy imports of heavy libs happen inside functions where possible,
# so that import time stays low and Streamlit can show progress.


# ---------- 1. Load & 2. Chunk ----------

@dataclass
class Chunk:
    """One retrievable unit of resume text."""
    section: str          # e.g. "Experience — Merck Group"
    text: str             # the chunk body
    chunk_id: int         # stable index into the chunk list


def load_resume(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def chunk_resume(raw: str, max_chars: int = 900) -> List[Chunk]:
    """
    Section-aware chunking.

    Resumes have natural boundaries (## headings, ### subheadings).
    We split on those first; if a section is still too long, we split
    further on sentence boundaries while keeping the heading as context.
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    # Split on ## or ### headers, keeping the header with its body.
    blocks = re.split(r"\n(?=## |### )", raw)

    current_h2: Optional[str] = None  # top-level section name

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Identify header
        first_line, _, body = block.partition("\n")
        first_line = first_line.strip()
        body = body.strip()

        if first_line.startswith("## "):
            current_h2 = first_line.lstrip("# ").strip()
            section_label = current_h2
            content = body
        elif first_line.startswith("### "):
            sub = first_line.lstrip("# ").strip()
            section_label = f"{current_h2} — {sub}" if current_h2 else sub
            content = body
        else:
            section_label = current_h2 or "General"
            content = block

        if not content:
            # Header-only block; still index the header itself so
            # questions like "what sections does the resume have" work.
            chunks.append(Chunk(section_label, first_line, chunk_id))
            chunk_id += 1
            continue

        # Split long sections into sentence-grouped sub-chunks.
        if len(content) <= max_chars:
            chunks.append(Chunk(section_label, f"{section_label}\n\n{content}", chunk_id))
            chunk_id += 1
        else:
            sentences = re.split(r"(?<=[.!?])\s+", content)
            buf: List[str] = []
            buf_len = 0
            for s in sentences:
                if buf_len + len(s) > max_chars and buf:
                    body_text = " ".join(buf).strip()
                    chunks.append(Chunk(section_label, f"{section_label}\n\n{body_text}", chunk_id))
                    chunk_id += 1
                    buf, buf_len = [s], len(s)
                else:
                    buf.append(s)
                    buf_len += len(s) + 1
            if buf:
                body_text = " ".join(buf).strip()
                chunks.append(Chunk(section_label, f"{section_label}\n\n{body_text}", chunk_id))
                chunk_id += 1

    return chunks


# ---------- 3. Embed & 4. Index ----------

class VectorIndex:
    """Thin wrapper around FAISS for cosine similarity search."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.index = None  # FAISS index, built in `build`
        self.dim: Optional[int] = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,   # so inner product == cosine sim
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")

    def build(self, chunks: List[Chunk]) -> None:
        import faiss
        self.chunks = chunks
        vecs = self._embed([c.text for c in chunks])
        self.dim = vecs.shape[1]
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

    def search(self, query: str, k: int = 4) -> List[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        qv = self._embed([query])
        scores, idxs = self.index.search(qv, k)
        results: List[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


# ---------- 5. Generate ----------

SYSTEM_PROMPT = """You are a friendly assistant answering questions about Krishnan Kannan's professional background, based ONLY on the resume excerpts provided below.

Rules:
- Ground every claim in the excerpts. If the answer isn't in the excerpts, say "That's not covered in my resume — feel free to ask me directly."
- Be concise. 2-4 sentences for most questions. Use bullets only for genuine lists (technologies, companies).
- Speak in third person about Krishnan ("Krishnan led...", "He architected...").
- Don't invent dates, employers, or metrics.
"""


def build_prompt(question: str, retrieved: List[tuple[Chunk, float]]) -> str:
    context_blocks = []
    for chunk, score in retrieved:
        context_blocks.append(f"[Source: {chunk.section}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"""Resume excerpts:

{context}

---

Question: {question}

Answer:"""


def generate_answer(
    question: str,
    retrieved: List[tuple[Chunk, float]],
    provider: str = "anthropic",
) -> str:
    """
    Calls the configured LLM. Provider is 'anthropic' or 'openai'.
    Reads API key from env: ANTHROPIC_API_KEY or OPENAI_API_KEY.
    """
    user_prompt = build_prompt(question, retrieved)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text.strip()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------- Convenience: one-shot builder ----------

def build_default_index(resume_path: str = "data/resume.txt") -> VectorIndex:
    raw = load_resume(resume_path)
    chunks = chunk_resume(raw)
    idx = VectorIndex()
    idx.build(chunks)
    return idx


if __name__ == "__main__":
    # Quick sanity check from the command line.
    idx = build_default_index()
    print(f"Indexed {len(idx.chunks)} chunks.\n")
    for q in [
        "What is Krishnan's experience with Snowflake?",
        "Has he worked in pharma?",
        "Which MDM domains has he mastered?",
    ]:
        print(f"Q: {q}")
        hits = idx.search(q, k=3)
        for c, s in hits:
            print(f"  [{s:.3f}] {c.section}")
        print()
