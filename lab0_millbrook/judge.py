"""LLM-as-judge over an eval question set (default: the 24 Fort Point questions).

For each question the naive baseline answers, then a second LLM call
compares that answer against the set's expected answer AND its list of
"wrong answers a RAG might give" (ready-made distractors). Verdicts:
correct / partially_correct / wrong / fabricated. Fabrication is called
out separately because the challenge rubric makes it a CRITICAL FAILURE
— a confident lie is worse than a miss.

Scoring bands (challenge rubric): EXCELLENT 90-100 / GOOD 75-89 /
FAIR 60-74 / POOR <60. Points: correct=1, partially_correct=0.5.

Run:  make lab0-score     (or: python -m lab0_millbrook.judge)

The original 22 Millbrook questions run via explicit --corpus-dir /
--questions / --collection flags — that's make lab0-millbrook-score.
The Makefile passes explicit flags for BOTH editions, so behavior never
depends on these defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from lab0_millbrook import naive_rag

QUESTIONS = naive_rag.REPO_ROOT / "lab0_boston" / "questions.json"  # Fort Point default
VERDICTS = ("correct", "partially_correct", "wrong", "fabricated")
POINTS = {"correct": 1.0, "partially_correct": 0.5, "wrong": 0.0, "fabricated": 0.0}

JUDGE_PROMPT = """You are grading a RAG system's answer. Reply with ONE word from:
correct, partially_correct, wrong, fabricated.

Use "fabricated" when the answer confidently states details that appear in
neither the expected answer nor the question (invented facts).

Question: {question}
Expected answer: {expected}
Known wrong answers a RAG might give: {wrong}
System's answer: {answer}

Verdict (one word):"""


def judge_one(llm, q: dict, answer: str) -> str:
    prompt = JUDGE_PROMPT.format(
        question=q["question"], expected=q["expected_answer"],
        wrong="; ".join(q["wrong_answers"]), answer=answer,
    )
    raw = llm.invoke(prompt).content.strip().lower()
    return next((v for v in VERDICTS if v in raw), "wrong")


def band(pct: float) -> str:
    if pct >= 90:
        return "EXCELLENT (90-100)"
    if pct >= 75:
        return "GOOD (75-89)"
    if pct >= 60:
        return "FAIR (60-74)"
    return "POOR (<60)"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge the naive baseline over an eval question set.")
    parser.add_argument("--corpus-dir", type=Path, default=naive_rag.DOCS_DIR,
                        help="split docs directory (default: Fort Point)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS,
                        help="questions.json path (default: Fort Point's 24)")
    parser.add_argument("--collection", default=naive_rag.COLLECTION,
                        help="chroma collection name (default: fortpoint_naive)")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    total_q = len(questions)
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=naive_rag.MODEL, temperature=0, num_ctx=naive_rag.NUM_CTX)
        llm.invoke("Reply with: ok")  # fail fast before a full run
    except Exception as exc:
        print(f"[!] Ollama is unreachable ({type(exc).__name__}): {exc}", file=sys.stderr)
        print(f"[!] Start it ('ollama serve') and pull: ollama pull {naive_rag.MODEL}", file=sys.stderr)
        raise SystemExit(1)

    by_cat: dict[str, list[str]] = defaultdict(list)
    fabricated: list[str] = []
    for i, q in enumerate(questions, 1):
        chunks = naive_rag.retrieve(q["question"], docs_dir=args.corpus_dir, collection=args.collection)
        answer = naive_rag.generate(q["question"], chunks) or "(no answer)"
        verdict = judge_one(llm, q, answer)
        by_cat[q["category"]].append(verdict)
        if verdict == "fabricated":
            fabricated.append(q["id"])
        print(f"  [{i:2d}/{total_q}] {q['id']:>2} {verdict:<17} {q['question'][:52]}")

    total = sum(POINTS[v] for vs in by_cat.values() for v in vs)
    pct = 100.0 * total / len(questions)
    print("\nPER-CATEGORY RESULTS")
    for cat, verdicts in sorted(by_cat.items()):
        score = sum(POINTS[v] for v in verdicts)
        print(f"  {cat:<40} {score:>4.1f} / {len(verdicts)}")
    print(f"\nTOTAL: {total:.1f} / {len(questions)}  ->  {pct:.0f}%   BAND: {band(pct)}")
    if fabricated:
        print(f"\nCRITICAL FAILURE: fabricated answers on {', '.join(fabricated)}")
        print("Per the rubric, fabricating information not in the documents is")
        print("an automatic critical failure — fix this before anything else.")
    print("\nYour goal Saturday is EXCELLENT — pick which failure you'll kill first.")


if __name__ == "__main__":
    main()
