import argparse
from .graph import build_app
from .ingestion import ingest_all

def ask(question):
    app = build_app()
    result = app.invoke({"question": question})
    print(f"Intent: {result.get('intent')}\nSource: {result.get('source')}\n\n Answer: {result.get('answer')}")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--question")
    parser.add_argument("--interactive", action="store_true", help="Start interactive Q&A loop")
    args = parser.parse_args()
    if args.ingest:
        print(ingest_all())
    elif args.question:
        ask(args.question)
    elif args.interactive:
        interactive()
    else:
        parser.error("Use --ingest or --question")

if __name__ == "__main__":
    main()