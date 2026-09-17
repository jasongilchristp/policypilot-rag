import argparse
import json
from pathlib import Path
from .graph import build_app
from .ingestion import ingest_all

def ask(question, json_out=False):
    app = build_app()
    result = app.invoke({"question": question})
    if json_out:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Intent: {result.get('intent')}\nSource: {result.get('source')}\n\nAnswer: {result.get('answer')}")

def interactive():
    app = build_app()
    print("Interactive mode. Type 'exit' or 'quit' to leave.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = app.invoke({"question": question})
        print(f"Intent: {result.get('intent')}\nSource: {result.get('source')}\n\nAnswer: {result.get('answer')}")

def run_eval(eval_file):
    from collections import defaultdict
    import json

    def load_eval(path):
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items

    FILE_TO_SOURCE = {
        "company_hr_policy.txt": ["hr", "hr_policy"],
        "engineering_standards.txt": ["engineering", "engineering_standards"],
        "onboarding_guide.txt": ["onboarding", "onboarding_guide"],
        "product_knowledge_base.txt": ["product", "product_knowledge", "product_knowledge_base"],
        "security_policy.txt": ["security", "security_policy"],
        None: ["general", "answer_general"]
    }

    def is_source_correct(gold_file, pred_source):
        if not pred_source:
            return 0
        pred = pred_source.lower()
        for opt in FILE_TO_SOURCE.get(gold_file, []):
            if opt in pred:
                return 1
        return 0

    def check_abstention(answer, should_answer):
        if not answer:
            return 0
        abstain_phrases = ["don't have", "not in the", "no information", "cannot find", "not found", "i don't know", "not provided"]
        abstained = any(p in answer.lower() for p in abstain_phrases)
        return 1 if abstained == (not should_answer) else 0

    def check_keyword(gold_keywords, answer):
        if not gold_keywords:
            return None
        if not answer:
            return 0
        ans_lower = answer.lower()
        for kw in gold_keywords:
            if kw.lower() in ans_lower:
                return 1
        return 0

    eval_path = Path(eval_file)
    if not eval_path.exists():
        # try /mnt/data fallback
        alt = Path("/mnt/data/eval_set.jsonl")
        if alt.exists():
            eval_path = alt
        else:
            print(f"Eval file not found: {eval_file}")
            return

    eval_items = load_eval(eval_path)
    app = build_app()
    print(f"Loaded {len(eval_items)} questions from {eval_path}")
    
    results = []
    for item in eval_items:
        out = app.invoke({"question": item["question"]})
        pred_intent = out.get("intent", "")
        pred_source = out.get("source", "")
        answer = out.get("answer", "")

        routing = 1 if pred_intent == item["gold_intent"] else 0
        source = is_source_correct(item["gold_file"], pred_source)
        keyword = check_keyword(item["gold_keywords"], answer)
        abstain = check_abstention(answer, item["should_answer"])
        results.append((routing, source, keyword, abstain))
        print(f"{item['gold_intent']:12} -> {pred_intent:12} | src {pred_source:20} | routing {routing} source {source}")

    routing_acc = sum(r[0] for r in results) / len(results)
    source_acc = sum(r[1] for r in results) / len(results)
    kw_scores = [r[2] for r in results if r[2] is not None]
    kw_acc = sum(kw_scores) / len(kw_scores) if kw_scores else 0
    abst_acc = sum(r[3] for r in results) / len(results)

    print("\n=== EVAL RESULTS ===")
    print(f"Routing Accuracy: {routing_acc:.2%}")
    print(f"Source Accuracy (right collection): {source_acc:.2%}")
    print(f"Keyword Recall: {kw_acc:.2%}")
    print(f"Abstention: {abst_acc:.2%}")

def main():
    parser = argparse.ArgumentParser(description="PolicyPilot RAG CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest docs into Chroma")
    parser.add_argument("--question", help="Ask one question")
    parser.add_argument("--interactive", action="store_true", help="Start interactive Q&A loop")
    parser.add_argument("--eval", nargs="?", const="eval/eval_set.jsonl", default=None, 
                        help="Run eval. Use --eval alone for default eval/eval_set.jsonl or --eval path/to/file.jsonl")
    parser.add_argument("--json", action="store_true", help="Output JSON for scripting")
    args = parser.parse_args()
    
    if args.ingest:
        print(ingest_all())
    elif args.question:
        ask(args.question, json_out=args.json)
    elif args.interactive:
        interactive()
    elif args.eval is not None:
        run_eval(args.eval)
    else:
        parser.error("Use --ingest or --question or --interactive or --eval")

if __name__ == "__main__":
    main()