
"""
PolicyPilot RAG - Eval Harness
How it maps to ML you know:

ML: y_true vs y_pred -> accuracy
RAG: 3 models in one -> 3 accuracies

1. CLASSIFY -> routing_accuracy (same as ML classification accuracy)
2. RETRIEVE -> recall@k (did right doc come back?)
3. GENERATE -> groundedness + abstention (is answer from context?)

Usage:
  python eval_harness.py --eval_file /mnt/data/eval_set.jsonl

It expects your graph to have:
  from company_support_rag.graph import build_graph
  graph.invoke({"question": "..."}) -> {"intent": "...", "answer": "...", "retrieved_docs": [...]}

If you haven't built that yet, it will run in DRY-RUN mode using gold data to show you metrics calculation.
"""

import json
import argparse
from collections import defaultdict

def load_eval(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def check_recall_at_k(gold_keywords, retrieved_texts, k=5):
    """Simple keyword-based recall@k - does gold keyword appear in top k docs?"""
    if not gold_keywords:
        return None
    combined = " ".join(retrieved_texts[:k]).lower()
    for kw in gold_keywords:
        if kw.lower() in combined:
            return 1
    return 0

def check_grounded_simple(answer, retrieved_texts):
    """For stable job, simple groundedness: at least one keyword from retrieved in answer"""
    if not answer or not retrieved_texts:
        return 0
    combined_retrieved = " ".join(retrieved_texts).lower()
    ans_words = set(answer.lower().split())
    ret_words = set(combined_retrieved.split())
    overlap = len(ans_words & ret_words)
    return 1 if overlap > 5 else 0

def check_abstention(answer, should_answer):
    """Should abstain when should_answer=False"""
    if not answer:
        return 0
    abstain_phrases = ["don't have", "not in the", "no information", "cannot find", "not found", "i don't know", "not provided"]
    abstained = any(p in answer.lower() for p in abstain_phrases)
    return 1 if abstained == (not should_answer) else 0

def run_eval(eval_file, dry_run=False):
    eval_items = load_eval(eval_file)
    print(f"Loaded {len(eval_items)} eval questions")
    
    graph = None
    if not dry_run:
        try:
            from company_support_rag.graph import build_graph
            graph = build_graph()
            print("Loaded real PolicyPilot graph")
        except Exception as e:
            print(f"Could not load graph ({e}), running in DRY-RUN mode")
            dry_run = True

    results = []
    for item in eval_items:
        q = item["question"]
        gold_intent = item["gold_intent"]
        gold_keywords = item["gold_keywords"]
        should_answer = item["should_answer"]
        
        if dry_run:
            pred_intent = gold_intent
            retrieved = [" ".join(gold_keywords)] if gold_keywords else []
            answer = f"Based on {item['gold_file']}, answer is {'; '.join(gold_keywords)}" if should_answer else "I don't have information about this in the company knowledge base."
        else:
            out = graph.invoke({"question": q})
            pred_intent = out.get("intent", "general")
            retrieved_docs = out.get("retrieved_docs", []) or out.get("context", [])
            if retrieved_docs and hasattr(retrieved_docs[0], 'page_content'):
                retrieved = [d.page_content for d in retrieved_docs]
            elif isinstance(retrieved_docs, list):
                retrieved = [str(d) for d in retrieved_docs]
            else:
                retrieved = [str(retrieved_docs)]
            answer = out.get("answer", "")

        routing_correct = 1 if pred_intent == gold_intent else 0
        recall = check_recall_at_k(gold_keywords, retrieved, k=5)
        grounded = check_grounded_simple(answer, retrieved) if should_answer else None
        abstain_correct = check_abstention(answer, should_answer)

        results.append({
            "question": q,
            "gold_intent": gold_intent,
            "pred_intent": pred_intent,
            "routing_correct": routing_correct,
            "recall@5": recall,
            "grounded": grounded,
            "abstain_correct": abstain_correct,
            "answer": answer[:200]
        })

    routing_acc = sum(r["routing_correct"] for r in results) / len(results)
    recall_scores = [r["recall@5"] for r in results if r["recall@5"] is not None]
    recall_5 = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    grounded_scores = [r["grounded"] for r in results if r["grounded"] is not None]
    grounded_acc = sum(grounded_scores) / len(grounded_scores) if grounded_scores else 0
    abstain_acc = sum(r["abstain_correct"] for r in results) / len(results)

    print("\n=== EVAL RESULTS (like ML classification report) ===")
    print(f"Routing Accuracy (intent classification): {routing_acc:.2%}")
    print(f"Recall@5 (retrieval): {recall_5:.2%}")
    print(f"Groundedness (is answer from context): {grounded_acc:.2%}")
    print(f"Abstention Accuracy (says IDK when should): {abstain_acc:.2%}")

    per_intent = defaultdict(list)
    for r in results:
        per_intent[r["gold_intent"]].append(r["routing_correct"])
    print("\nPer-intent routing:")
    for intent, scores in per_intent.items():
        print(f"  {intent}: {sum(scores)/len(scores):.2%} ({len(scores)} qs)")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default="/mnt/data/eval_set.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_eval(args.eval_file, dry_run=args.dry_run)
