PRECISE_PROMPT_TEMPLATE = """You are an assistant for UAV research.
Use ONLY the provided context to answer the question in 4-6 sentences.
If the context does not contain the answer, say: 'I do not know based on the provided papers.'

Question:
{question}

Context:
{context}
"""

STRUCTURED_PROMPT_TEMPLATE = """You are an assistant for UAV research.
Use ONLY the context and provide:
1) A direct answer
2) Key evidence points (bullet list)
3) Mention gaps if evidence is incomplete

Question:
{question}

Context:
{context}
"""


def build_prompt(style: str, question: str, context: str) -> str:
    templates = {
        "precise": PRECISE_PROMPT_TEMPLATE,
        "structured": STRUCTURED_PROMPT_TEMPLATE,
    }
    template = templates.get(style, PRECISE_PROMPT_TEMPLATE)
    return template.format(question=question, context=context)
