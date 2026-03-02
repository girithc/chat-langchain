# Intent monitor agent — classifies conversation deployability in parallel with docs agent
import logging

from langchain.agents import create_agent

from src.agent.config import (
    configurable_model,
    model_fallback_middleware,
)
from src.prompts.monitor_prompt import monitor_agent_prompt

logger = logging.getLogger(__name__)

intent_monitor = create_agent(
    model=configurable_model,
    tools=[],  # No tools needed — pure classification
    system_prompt=monitor_agent_prompt,
    middleware=[
        model_fallback_middleware,
    ],
)

logger.info("Intent monitor agent loaded")
