# LangGraph Sample Agent

A minimal LangGraph agent you can deploy on LangSmith. This folder is self-contained.

## Files

- `graph.py` defines a tiny chat graph.
- `langgraph.json` is the LangGraph config for this sample.

## Run Locally

1. `cd agent/langchain/langgraph`
2. Create a `.env` with your model API key (for example `OPENAI_API_KEY`).
3. (Optional) Set `LANGGRAPH_SAMPLE_MODEL` (default: `openai:gpt-4o-mini`).
4. Run `langgraph dev`

## Deploy on LangSmith

1. Push your repo to GitHub.
2. In LangSmith, create a new deployment and point it at `agent/langchain/langgraph/langgraph.json`.
3. Add your API key and `LANGGRAPH_SAMPLE_MODEL` in the deployment environment.
4. Deploy.
