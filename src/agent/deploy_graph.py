# Deploy agent — generates LangGraph code scaffolds from conversation context
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from src.prompts.deploy_prompt import deploy_agent_prompt

logger = logging.getLogger(__name__)

# Use a capable model with JSON mode enforced to guarantee valid JSON output
model = init_chat_model(
    model="openai:gpt-4o",
    model_kwargs={"response_format": {"type": "json_object"}},
)


def generate_code(state: MessagesState):
    """Call the model with the deploy system prompt and return its response."""
    messages = [SystemMessage(content=deploy_agent_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("generate_code", generate_code)
builder.add_edge(START, "generate_code")
builder.add_edge("generate_code", END)

deploy_agent = builder.compile()

logger.info("Deploy agent loaded")
