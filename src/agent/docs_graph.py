# Docs agent for LangChain customer service with docs and knowledge base tools
import logging

from langgraph.prebuilt import create_react_agent

from src.agent.config import (
    GUARDRAILS_MODEL,
    configurable_model,
)
from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools.docs_tools import SearchDocsByLangChain
from src.tools.link_check_tools import check_links
from src.tools.pylon_tools import get_article_content, search_support_articles

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.info("Docs agent module loaded")

docs_agent = create_react_agent(
    model=configurable_model,
    tools=[
        SearchDocsByLangChain,
        search_support_articles,
        get_article_content,
        check_links,
    ],
    prompt=docs_agent_prompt,
)
