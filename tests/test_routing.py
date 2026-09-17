import pytest
from company_support_rag.graph import build_app

# Your current 96% eval - this locks it so you never regress
GOLD = [
    ("How many days of paid annual leave do full-time employees get per year?", "hr"),
    ("What is the branch naming convention for new features?", "engineering"),
    ("What time should new employees report on Day 1 and what ID to bring?", "onboarding"),
    ("What is the pricing for Starter plan billed annually?", "product"),
    ("What are the password requirements for minimum length?", "security"),
    ("What is the capital of France?", "general"),
    ("Which WiFi network should employees use in office?", "security"),
    ("Can we paste customer data into ChatGPT or Claude?", "security"),
    ("Is WFH available during probation period?", "onboarding"),
    ("What is the maximum allowed query time for API endpoints?", "engineering"),
]

def test_routing_accuracy():
    app = build_app()
    correct = 0
    for q, gold_intent in GOLD:
        result = app.invoke({"question": q})
        pred = result.get("intent", "")
        if pred == gold_intent:
            correct += 1
        else:
            print(f"FAIL: {q} -> got {pred} expected {gold_intent}")
    acc = correct / len(GOLD)
    print(f"Routing accuracy: {acc:.0%} ({correct}/{len(GOLD)})")
    assert acc >= 0.9, f"Routing dropped to {acc:.0%}, expected >=90%"

def test_routes_defaults_to_general():
    app = build_app()
    result = app.invoke({"question": "What is the capital of Japan?"})
    assert result.get("intent") == "general"

def test_abstention_for_general():
    app = build_app()
    result = app.invoke({"question": "Write a poem about monsoon."})
    answer = result.get("answer", "").lower()
    # Should contain abstention phrase
    assert any(p in answer for p in ["don't have", "no information", "not in the"]), f"Did not abstain: {answer}"
