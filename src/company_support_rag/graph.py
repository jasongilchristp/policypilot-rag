from langgraph.graph import StateGraph, START, END
from .schemas import SupportState
from .nodes import *

def build_app():
    graph = StateGraph(SupportState)
    graph.add_node("classify", classify_intent)

    for name, node in [
        ("hr_policy", search_hr_policy),
        ("engineering_standards", search_engineering_standards),
        ("onboarding_guide", search_onboarding_guide),
        ("product_knowledge", search_product_knowledge_base),
        ("security_policy", search_security_policy),
        ("answer_general", answer_general),
        ("generate", generate_answer),
    ]:
        graph.add_node(name, node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {
            "hr": "hr_policy",
            "engineering": "engineering_standards",
            "onboarding": "onboarding_guide",
            "product": "product_knowledge",
            "security": "security_policy",
            "general": "answer_general",
        },
    )

    for name in (
        "hr_policy",
        "engineering_standards",
        "onboarding_guide",
        "product_knowledge",
        "security_policy",
    ):
        graph.add_edge(name, "generate")

    graph.add_edge("answer_general", END)
    graph.add_edge("generate", END)

    return graph.compile()
