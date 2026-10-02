import time

from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .models import get_chat_model, get_intent_model
from .prompts import ABSTENTION_MESSAGE, ANSWER_SYSTEM, GENERAL_SYSTEM, INTENT_SYSTEM
from .retrieval import retrieve_from_chroma, format_context, route_by_similarity

VALID_INTENTS = ["hr", "engineering", "onboarding", "product", "security", "general"]

# Groq's free tier enforces a low tokens-per-minute cap, so bursts of concurrent
# requests fail with 429 even when total volume is modest. Back off and retry
# rather than crashing the run.
_RATE_LIMIT_BACKOFF_SECONDS = (5, 10, 20, 40, 60, 60)

# Small models sometimes emit typographic Unicode (for example U+202F narrow
# no-break space) inside copied values like "Rs 299". That breaks exact keyword
# and phrase matching even though the answer reads correctly, so normalise the
# output instead of trusting the model to avoid those characters.
_TYPOGRAPHIC_MAP = str.maketrans({
    " ": " ",   # no-break space
    " ": " ",   # narrow no-break space
    " ": " ",   # thin space
    " ": " ",   # figure space
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "‐": "-",   # hyphen
    "‑": "-",   # non-breaking hyphen
    "‒": "-",   # figure dash
    "–": "-",   # en dash
    "—": "-",   # em dash
})


def _normalize(text: str) -> str:
    return (text or "").translate(_TYPOGRAPHIC_MAP).strip()


def _is_rate_limit(exc: Exception) -> bool:
    return "RateLimit" in type(exc).__name__ or "429" in str(exc)


def _invoke_with_retry(model, system, user) -> str:
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    last_error: Exception | None = None
    for wait in (0, *_RATE_LIMIT_BACKOFF_SECONDS):
        if wait:
            print(f"rate limited, retrying in {wait}s", flush=True)
            time.sleep(wait)
        try:
            return _normalize(model.invoke(messages).content)
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            last_error = exc
    raise last_error


def call_llm(system, user):
    return _invoke_with_retry(get_chat_model(), system, user)


def call_intent(system, user):
    return _invoke_with_retry(get_intent_model(), system, user)


def classify_intent(state):
    question = state["question"]

    # Embedding router first: it is deterministic, costs no prompt tokens, and
    # generalises to phrasings no prompt rule was written for.
    intent, _distance = route_by_similarity(question)
    if intent is not None:
        return {"intent": intent}

    raw = call_intent(INTENT_SYSTEM, f"Question: {question}").lower()
    # handles "hr." or " hr " or "Answer: hr"
    cleaned = raw.split()[0].strip(".,:()[]\"'") if raw else "general"
    if cleaned not in VALID_INTENTS:
        # search inside: "the answer is hr" -> finds hr
        for label in VALID_INTENTS:
            if label in raw:
                return {"intent": label}
        return {"intent": "general"}
    return {"intent": cleaned}


def search_collection(state, intent):
    docs = retrieve_from_chroma(state["question"], settings.collections[state["intent"]])
    return {"context": format_context(docs), "source": settings.collections[state["intent"]]}


def search_hr_policy(state): return search_collection(state, "hr")
def search_engineering_standards(state): return search_collection(state, "engineering")
def search_onboarding_guide(state): return search_collection(state, "onboarding")
def search_product_knowledge_base(state): return search_collection(state, "product")
def search_security_policy(state): return search_collection(state, "security")


def generate_answer(state):
    context = state.get("context", "")
    if not context.strip():
        # Empty retrieval is an abstention and must use the shared wording so the
        # eval's abstention check recognises it.
        return {"answer": ABSTENTION_MESSAGE}
    answer = call_llm(
        ANSWER_SYSTEM,
        f"Question: {state['question']}\n\nContext:\n{context}"
    )
    return {"answer": answer}


def is_abstention(answer: str) -> bool:
    """True when the model refused rather than answered.

    Used to fall back to the canonical wording when the model paraphrased a
    refusal, so callers see one stable string.
    """
    lowered = (answer or "").lower()
    markers = (
        "don't have information",
        "do not have information",
        "no information",
        "not in the context",
        "not mentioned",
        "cannot find",
        "can't find",
    )
    return any(marker in lowered for marker in markers)


def answer_general(state): return {"answer": call_llm(GENERAL_SYSTEM, state["question"]), "source": "general", "context": ""}
def route(state): return state.get("intent", "general") if state.get("intent", "general") in VALID_INTENTS else "general"