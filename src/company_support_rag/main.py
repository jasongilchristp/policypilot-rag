import argparse
from .graph import build_app
from .ingestion import ingest_all

def ask(question):
    result = build_app.invoke({"question": question})
    print(f"Intent: {result.get('intent')}\nSource: {result.get('source')}\n\n Answer: {result.get('answer')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--question")
    args = parser.parse_args()
    if args.ingest:
        print(ingest_all())
    elif args.question:
        ask(args.question)
    else:
        parser.error("Use --ingest or --question")

if __name__ == "__main__":
    main()