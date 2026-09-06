from langchain_core.messages import HumanMessage, SystemMessage
from .config import settings
from .models import get_llm
from .prompts import ANSWER_SYSTEM, GENERAL_SYSTEM, INTENT_SYSTEM
from .retrieval import retrieve_from_chroma, format_context

VALID_INTENTS = {"hr", "engineering", "onboarding", "product", "security", "general"}

def call_llm(system, user):
    return get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip()

def classify_intent(state):
    value = call_llm(INTENT_SYSTEM, f"Question: {state['question']}").lower()
    return {"intent": value if value in VALID_INTENTS else "general"}

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
        return {"answer": "No relevant documents found."}
    answer = call_llm(
        ANSWER_SYSTEM,
        f"Question: {state['question']}\n\nContext:\n{context}"
    )
    return {"answer": answer}

def answer_general(state): return {"answer": call_llm(GENERAL_SYSTEM, state["question"]), "source": "general", "context": ""}
def route(state): return state.get("intent", "general") if state.get("intent", "general") in VALID_INTENTS else "general"