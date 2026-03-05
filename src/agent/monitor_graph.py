# Intent monitor agent — classifies conversation deployability in parallel with docs agent
import logging

from langgraph.prebuilt import create_react_agent

from src.agent.config import (
    configurable_model,
)
from src.prompts.monitor_prompt import monitor_agent_prompt

logger = logging.getLogger(__name__)

intent_monitor = create_react_agent(
    model=configurable_model,
    tools=[],  # No tools needed — pure classification
    prompt=monitor_agent_prompt,
)

logger.info("Intent monitor agent loaded")
