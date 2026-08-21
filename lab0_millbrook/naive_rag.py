"""The deliberately naive RAG baseline. Every choice here is a lesson.

Naive on purpose:
  * fixed ~500-token chunks, no overlap — chunk borders cut entities and
    dates in half, so answers that need two halves never see both;
  * chromadb's default embedder (all-MiniLM-L6-v2) — small, monolingual,
    and blind to the non-English spans in the transcripts (Spanish /
    Haitian Creole / Chinese in Fort Point; Arabic / Spanish / Mandarin
    in Millbrook);
  * top-3 similarity, no reranking, no metadata filters — whatever
    LOOKS like the question wins, and similarity is not identity;
  * one LLM call over the raw chunks — no date arithmetic, no entity
    graph, no contradiction check, no abstention logic.

We print the retrieved chunks BEFORE the answer so that when the model
is wrong, you can see exactly which chunk lied to it.

CLI:  python -m lab0_millbrook.naive_rag "Which Peña spoke at the May 19 committee meeting?"

Defaults point at Lab 0 — The Fort Point Files (lab0_boston). The
original Millbrook corpus runs via explicit --corpus-dir / --collection
flags; the Makefile passes explicit flags for BOTH editions, so behavior
never depends on these defaults. Each edition gets its own chroma
collection (fortpoint_naive / millbrook_naive), so the indexes never
collide.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "lab0_boston" / "corpus" / "docs"  # Fort Point default
CHROMA_DIR = str(REPO_ROOT / ".chroma" / "lab0")
COLLECTION = "fortpoint_naive"  # Millbrook uses millbrook_naive via --collection
CHUNK_TOKENS = 500  # fixed size, no overlap — deliberately naive
TOP_K = 3
MODEL = os.environ.get("OLLAMA_MODEL", "granite3.1-dense:8b")
# Ollama defaults this model to a 32768-token window, which reserves ~13GB and spills
# off a 16GB GPU (measured 0.5 tok/s). 16384 fits entirely in GPU at 9.5GB -> ~9.4 tok/s,
# ~19x faster, with ample room for larger top-k. Raise only if context actually overflows.
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

PROMPT = (
    "Answer the question using ONLY the context below.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def _chunk(text: str, size: int = CHUNK_TOKENS) -> list[str]:
    # "Tokens" approximated by whitespace words — naive baselines don't
    # even own a tokenizer. Boundaries land wherever they land.
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def build_index(docs_dir: Path = DOCS_DIR, collection: str = COLLECTION) -> chromadb.Collection:
    """Index docs_dir into a persistent chroma collection (idempotent)."""
    if not any(docs_dir.glob("doc*.md")):
        from lab0_millbrook.split_corpus import FULL_FOR_DOCS, split
        full = FULL_FOR_DOCS.get(Path(docs_dir).resolve())
        if full is not None:  # a known edition — split its corpus in place
            split(full, docs_dir)
        else:
            raise SystemExit(
                f"No doc*.md files in {docs_dir} — run the splitter first:\n"
                f"  python -m lab0_millbrook.split_corpus --input <corpus.md> --outdir {docs_dir}"
            )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_or_create_collection(collection)
    if coll.count() > 0:
        return coll
    ids, docs, metas = [], [], []
    for path in sorted(docs_dir.glob("doc*.md")):
        for i, chunk in enumerate(_chunk(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metas.append({"source": path.name, "chunk": i})
    coll.add(ids=ids, documents=docs, metadatas=metas)  # default embedder
    return coll


def retrieve(question: str, k: int = TOP_K, docs_dir: Path = DOCS_DIR,
             collection: str = COLLECTION) -> list[dict]:
    coll = build_index(docs_dir, collection)
    res = coll.query(query_texts=[question], n_results=k)
    return [
        {"id": i, "text": d, "source": m["source"]}
        for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]


def generate(question: str, chunks: list[dict]) -> str | None:
    """One LLM call, or None (with a helpful message) if Ollama is down."""
    context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=MODEL, temperature=0, num_ctx=NUM_CTX)
        return llm.invoke(PROMPT.format(context=context, question=question)).content
    except Exception as exc:  # connection refused, model not pulled, ...
        print(f"\n[!] Could not reach Ollama ({type(exc).__name__}): {exc}", file=sys.stderr)
        print(f"[!] Fix: start Ollama ('ollama serve') and pull the model:", file=sys.stderr)
        print(f"[!]      ollama pull {MODEL}", file=sys.stderr)
        print("[!] Retrieval still ran — the chunks above are real.", file=sys.stderr)
        return None


def ask(question: str, docs_dir: Path = DOCS_DIR,
        collection: str = COLLECTION) -> tuple[list[dict], str | None]:
    """Retrieve, print the evidence, then answer. Returns (chunks, answer)."""
    chunks = retrieve(question, docs_dir=docs_dir, collection=collection)
    print(f"\nQ: {question}")
    print(f"-- retrieved top {len(chunks)} chunks " + "-" * 30)
    for n, c in enumerate(chunks, 1):
        preview = c["text"][:240].replace("\n", " ")
        print(f"  [{n}] {c['source']}  ::  {preview}...")
    answer = generate(question, chunks)
    if answer is not None:
        print("-- answer " + "-" * 48)
        print(answer.strip())
    return chunks, answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the deliberately naive RAG baseline one question.")
    parser.add_argument("--corpus-dir", type=Path, default=DOCS_DIR, help="split docs directory (default: Fort Point)")
    parser.add_argument("--collection", default=COLLECTION, help="chroma collection name (default: fortpoint_naive)")
    parser.add_argument("question", nargs="*", help="the question to ask")
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit('Usage: python -m lab0_millbrook.naive_rag "your question"')
    ask(question, docs_dir=args.corpus_dir, collection=args.collection)


if __name__ == "__main__":
    main()
