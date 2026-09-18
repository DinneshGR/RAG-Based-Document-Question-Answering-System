"""
Lightweight, fully local evaluation script for the RAG pipeline.

This script does NOT invent numbers: every metric is computed from the
system's actual behavior against the questions in dataset.json. Before
running this, make sure:

1. FastAPI backend + Ollama are running.
2. The document(s) referenced in dataset.json's "relevant_document"
   field have actually been uploaded/indexed (see README for a sample
   document + matching dataset).

Usage:
    python evaluate.py
    python evaluate.py --api-url http://localhost:8000

Metrics computed (all local, no paid LLM judge required):

* Retrieval Recall@K       - was the expected document/page among the
                              top-K retrieved chunks?
* Context Relevance        - fraction of expected keywords that appear
                              in the retrieved context (proxy for
                              "did retrieval find relevant material").
* Faithfulness/Grounding   - fraction of the generated answer's content
                              words that also appear somewhere in the
                              retrieved context (proxy for hallucination
                              rate; higher = more grounded).
* Answer Accuracy          - fraction of expected keywords present in
                              the generated answer (simple lexical
                              proxy for correctness; no paid LLM judge).

These are intentionally simple, transparent, local metrics rather than
an LLM-as-judge (which would require another model/API). This is noted
explicitly as a limitation in the README.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to",
    "and", "or", "for", "with", "this", "that", "it", "as", "at", "by",
    "be", "has", "have", "had", "what", "who", "when", "where", "how",
}


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def evaluate(api_url: str, dataset_path: Path, top_k: int = 5) -> None:
    cases = load_dataset(dataset_path)
    if not cases:
        print("No evaluation cases found in dataset.json.")
        return

    recall_hits = 0
    context_relevance_scores = []
    faithfulness_scores = []
    accuracy_scores = []
    skipped = 0

    for case in cases:
        question = case["question"]
        expected_doc = case.get("relevant_document")
        expected_page = case.get("relevant_page")
        expected_keywords = [k.lower() for k in case.get("expected_keywords", [])]

        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"question": question, "top_k": top_k, "mode": "mmr"},
                timeout=180,
            )
        except requests.exceptions.RequestException as exc:
            print(f"[SKIPPED] '{question}' -> could not reach API: {exc}")
            skipped += 1
            continue

        if resp.status_code != 200:
            print(f"[SKIPPED] '{question}' -> API error: {resp.json().get('detail', resp.text)}")
            skipped += 1
            continue

        data = resp.json()
        answer = data["answer"]
        passages = data["retrieved_passages"]
        context_text = " ".join(p["text"] for p in passages).lower()

        # --- Retrieval Recall@K ---
        hit = any(
            p["document_name"] == expected_doc
            and (expected_page is None or p.get("page_number") == expected_page)
            for p in passages
        )
        recall_hits += int(hit)

        # --- Context Relevance (keyword coverage in retrieved context) ---
        if expected_keywords:
            found = sum(1 for kw in expected_keywords if kw in context_text)
            context_relevance_scores.append(found / len(expected_keywords))

        # --- Faithfulness (answer words grounded in retrieved context) ---
        answer_tokens = set(tokenize(answer))
        context_tokens = set(tokenize(context_text))
        if answer_tokens:
            grounded = len(answer_tokens & context_tokens)
            faithfulness_scores.append(grounded / len(answer_tokens))

        # --- Answer Accuracy (expected keyword coverage in the answer) ---
        if expected_keywords:
            answer_lower = answer.lower()
            found = sum(1 for kw in expected_keywords if kw in answer_lower)
            accuracy_scores.append(found / len(expected_keywords))

        print(f"[OK] '{question}' -> retrieval_hit={hit}")

    evaluated = len(cases) - skipped
    if evaluated == 0:
        print("\nNo cases could be evaluated (all requests failed). "
              "Check that the backend/Ollama are running and documents are indexed.")
        return

    def avg(lst: List[float]) -> float:
        return round(100 * sum(lst) / len(lst), 2) if lst else 0.0

    print("\n===== Evaluation Results =====")
    print(f"Cases evaluated: {evaluated} / {len(cases)} (skipped: {skipped})")
    print(f"Retrieval Recall@{top_k}: {round(100 * recall_hits / evaluated, 2)}%")
    print(f"Average Context Relevance: {avg(context_relevance_scores)}%")
    print(f"Faithfulness/Grounding: {avg(faithfulness_scores)}%")
    print(f"Answer Accuracy (keyword-based): {avg(accuracy_scores)}%")
    print("\nNote: Context Relevance and Answer Accuracy use simple keyword-overlap "
          "heuristics rather than an LLM judge, since this project uses only free, "
          "local tools. See README 'Evaluation methodology' for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline locally.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI base URL")
    parser.add_argument(
        "--dataset", default=str(Path(__file__).parent / "dataset.json"), help="Path to dataset.json"
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.api_url, Path(args.dataset), args.top_k)
