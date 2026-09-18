"""Unit tests for app.llm.prompts."""
from app.llm.prompts import build_rag_prompt, format_context


def test_format_context_includes_source_and_page():
    chunks = [
        {"text": "Revenue grew by 18%.", "document_name": "report.pdf", "page_number": 12},
        {"text": "Costs decreased slightly.", "document_name": "report.pdf", "page_number": 14},
    ]
    context = format_context(chunks)
    assert "report.pdf" in context
    assert "page 12" in context
    assert "page 14" in context
    assert "Revenue grew by 18%." in context


def test_build_rag_prompt_structure():
    chunks = [{"text": "Some fact.", "document_name": "doc.txt", "page_number": None}]
    prompt = build_rag_prompt("What is the fact?", chunks)

    assert "system" in prompt and "user" in prompt
    assert "only the provided context" in prompt["system"] or "only the retrieved context" in prompt["system"]
    assert "What is the fact?" in prompt["user"]
    assert "Some fact." in prompt["user"]


def test_prompt_instructs_fallback_on_missing_info():
    chunks = [{"text": "Irrelevant.", "document_name": "doc.txt", "page_number": 1}]
    prompt = build_rag_prompt("Unrelated question?", chunks)
    assert "I couldn't find this information in the uploaded documents." in prompt["system"]
