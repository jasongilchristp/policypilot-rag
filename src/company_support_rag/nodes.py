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
    docs = retrieve_from_chroma(state["question"], settings.collections["intent"])
    return {"context": format_context(docs), "source": settings.collections[intent]}

def search_hr_policy(s): return search_collection(s, "hr")
def search_engineering_standards(s): return search_collection(s, "engineering")
def search_onboarding_guide(s): return search_collection(s, "onboarding")
def search_product_knowledge_base(s): return search_collection(s, "product")
def search_security_policy(s): return search_collection(s, "security")

def generate_answer(s): return {"answer": call_llm(ANSWER_SYSTEM, f"Context:\n{s.get('context', '')}\n\n Question:\n{s['question']}")}
def answer_general(s): return {"answer": call_llm(GENERAL_SYSTEM, s["question"]), "source": "general", "context": ""}
def route(s): return s.get("intent", "general") if s.get("intent", "general") in VALID_INTENTS else "general"