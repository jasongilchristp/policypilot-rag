from company_support_rag import route

def test_routes_defaults_to_general():
    assert route({}) == "general"

def test_routes_rejects_unknown_intent():
    assert route({"intent": "finance"}) == "general"