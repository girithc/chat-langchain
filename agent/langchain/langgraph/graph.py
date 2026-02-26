"""Minimal LangGraph agent for LangSmith deployment demos."""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


MODEL_ID = os.getenv("LANGGRAPH_SAMPLE_MODEL", "openai:gpt-4o-mini")


def assistant(state: State) -> dict:
    """Run a single LLM call and append the response to messages."""
    model = init_chat_model(model=MODEL_ID)
    response = model.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.set_entry_point("assistant")
builder.add_edge("assistant", END)

graph = builder.compile()
