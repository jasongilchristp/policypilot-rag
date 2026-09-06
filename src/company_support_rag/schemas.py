from typing import TypedDict

class SupportState(TypedDict, total=False):
    question: str
    intent: str
    context: str
    answer: str
    source: str