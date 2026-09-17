import json
from pathlib import Path

def test_eval_file_valid():
    p = Path("eval/eval_set.jsonl")
    assert p.exists(), "eval/eval_set.jsonl missing"
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 25, f"Expected 25 questions, got {len(lines)}"
    for line in lines:
        item = json.loads(line)
        assert "question" in item
        assert "gold_intent" in item
        assert "should_answer" in item
