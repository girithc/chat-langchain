"""Sandbox agent — orchestrates E2B sandbox creation, file writing, and execution.

Registered as 'sandbox_agent' in langgraph.json.
The frontend POSTs a JSON action message; this agent parses it,
calls the E2B sandbox utilities, and returns structured results.
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState

from src.tools.sandbox_tools import create_and_run

logger = logging.getLogger(__name__)


def sandbox_node(state: MessagesState) -> MessagesState:
    """Process a sandbox action request and return results."""
    messages = state.get("messages", [])

    # Find the last human message (LangChain message objects)
    human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            human_msg = msg
            break
        # Also handle dict-style messages
        if isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            human_msg = msg
            break

    if not human_msg:
        return {
            "messages": [AIMessage(content=json.dumps({"error": "No input message received"}))],
        }

    # Extract content
    if isinstance(human_msg, dict):
        content = human_msg.get("content", "")
    else:
        content = human_msg.content

    # Parse the action from the message content
    try:
        action_data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {
            "messages": [AIMessage(content=json.dumps({"error": f"Invalid JSON input: {str(content)[:200]}"}))],
        }

    action = action_data.get("action", "unknown")
    logger.info(f"Sandbox agent action: {action}")

    if action == "create_and_run":
        files = action_data.get("files", [])
        if not files:
            result = {"error": "No files provided"}
        else:
            result = create_and_run(files)
    else:
        result = {"error": f"Unknown action: {action}"}

    return {
        "messages": [AIMessage(content=json.dumps(result))],
    }


# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("sandbox", sandbox_node)
graph_builder.add_edge(START, "sandbox")
graph_builder.add_edge("sandbox", END)

sandbox_agent = graph_builder.compile()

logger.info("Sandbox agent loaded")
