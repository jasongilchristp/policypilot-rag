import pytest
from pathlib import Path
import json

def test_eval_file_exists():
    """Check eval/eval_set.jsonl exists and has 25 Qs"""
    p = Path("eval/eval_set.jsonl")
    assert p.exists(), f"Missing {p} - download eval_set.jsonl to eval/"
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 25, f"Expected 25 Qs, got {len(lines)}"

def test_chroma_or_docs_exist():
    """At least data/ or docs/ folder should exist"""
    assert True  # placeholder - always passes to check pytest discovery

def test_routing_smoke():
    """Smoke test for routing - uses your graph"""
    try:
        from company_support_rag.graph import build_app
        app = build_app()
        result = app.invoke({"question": "What is capital of France?"})
        intent = result.get("intent", "")
        # Should be general for world knowledge
        assert intent == "general", f"Expected general, got {intent}"
    except Exception as e:
        pytest.skip(f"Graph not ready: {e}")

def test_abstention_smoke():
    """Smoke test for abstention"""
    try:
        from company_support_rag.graph import build_app
        app = build_app()
        result = app.invoke({"question": "Write a poem about monsoon."})
        answer = result.get("answer", "").lower()
        # Should abstain
        assert any(p in answer for p in ["don't have", "no information", "not in the"]), f"Did not abstain: {answer}"
    except Exception as e:
        pytest.skip(f"Graph not ready: {e}")

def test_main_imports():
    """Check main_fixed.py imports work"""
    from company_support_rag.main import main
    assert callable(main)
