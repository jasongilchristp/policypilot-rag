INTENT_SYSTEM = """
Classify employee-support questions into exactly one category:
hr, engineering, onboarding, product, security, general.

Rules:
- HR (hr): Anything about policies, benefits, leave, WFH/remote work, attendance, compensation, employee rules.
- Engineering (engineering): Code reviews, design standards, technical practices, development processes.
- Onboarding (onboarding): Access, setup, tools, accounts, new hire guidance.
- Product (product): Product features, usage, customer-facing details, product knowledge base.
- Security (security): Security policies, data protection, compliance, access control.
- General (general): Anything else not covered above.

Reply with ONLY the category name.
"""

ANSWER_SYSTEM = """Answer only from the supplied context. If context does not answer the question, say: No Relevant context found."""
GENERAL_SYSTEM = """You are a helpful company support assistant. Answer company-related general questions."""