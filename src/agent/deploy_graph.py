# Deploy agent — generates LangGraph code scaffolds from conversation context
import json
import logging

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from src.prompts.deploy_prompt import deploy_agent_prompt

logger = logging.getLogger(__name__)

# Use a capable model for code generation
model = init_chat_model(
    model="openai:gpt-4o-mini",
    configurable_fields=("model",),
)

deploy_agent = create_agent(
    model=model,
    tools=[],  # No tools needed — pure LLM code generation
    system_prompt=deploy_agent_prompt,
)

logger.info("Deploy agent loaded")
