"""Lightweight conversation-context check for the boardgame rules agent.

Runs as the first node in the agent graph. Its only job is to detect when
the answer is already present in the conversation history (follow-up
clarifications, rephrasing of a question just answered, etc.) so the agent
can skip retrieval entirely.

All actual reasoning about complexity, cross-referencing, and gap-filling
happens in the ReAct loop via the system prompt — not here.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from boardgame_agent.agent.state import AgentState

_CHECK_PROMPT = """\
You are a pre-check for a board game rules assistant. Given the conversation \
so far, decide ONLY whether the user's latest question can be answered from \
information already present in the conversation (prior answers).

Rules:
- Answer YES only if the prior assistant messages CLEARLY contain the answer \
to the new question. This includes rephrased questions, follow-up clarifications \
about something just discussed, or "can you repeat that?"
- Answer NO if the question asks about ANYTHING not already covered, even if \
it seems related to a prior answer.
- When in doubt, answer NO — it's better to search than to miss information.

Respond with ONLY valid JSON (no markdown fencing):
{"skip_retrieval": true|false}"""


def classify_and_plan(
    state: AgentState,
    llm: Any,
    has_glossary: bool = False,
) -> dict:
    """Check if the answer is already in conversation context.

    Returns ``{"plan": ["skip"]}`` if retrieval can be skipped,
    or ``{"plan": None}`` to proceed with normal retrieval.
    """
    messages = state["messages"]

    # If this is the first message, always retrieve.
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if len(user_messages) <= 1:
        return {"plan": None}

    # Build a compact conversation summary for the check.
    context_lines: list[str] = []
    for msg in messages[-8:]:  # Last 4 turns max
        if isinstance(msg, HumanMessage):
            context_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            context_lines.append(f"Assistant: {msg.content[:400]}")

    if not context_lines:
        return {"plan": None}

    latest_question = user_messages[-1].content
    conversation = "\n".join(context_lines)

    response = llm.invoke([
        SystemMessage(content=_CHECK_PROMPT),
        HumanMessage(content=f"Conversation:\n{conversation}\n\nLatest question: {latest_question}"),
    ])

    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        if data.get("skip_retrieval") is True:
            return {"plan": ["Answer directly from conversation context — no search needed."]}
    except (json.JSONDecodeError, AttributeError):
        pass

    return {"plan": None}
