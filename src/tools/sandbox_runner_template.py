import json
import os
import sys
import time

import uuid
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"

from langsmith import Client

# Add the project dir to Python path so we can import the agent
sys.path.insert(0, "/home/user/project")

# Unique run name to identify this specific execution in LangSmith
run_name = f"chat_langchain_{uuid.uuid4().hex[:8]}"
os.environ["LANGCHAIN_RUN_NAME"] = run_name

# Keep project env vars aligned for both legacy and current SDK conventions.
if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")


def _safe_id(value):
    return str(value) if value is not None else None


def _to_iso(ts):
    if ts is None:
        return None
    try:
        return ts.isoformat()
    except Exception:
        return str(ts)


def _latency_ms(start_time, end_time):
    if start_time is None or end_time is None:
        return None
    try:
        return int((end_time - start_time).total_seconds() * 1000)
    except Exception:
        return None


def _dict_keys(value, limit=8):
    if isinstance(value, dict):
        return [str(k) for k in list(value.keys())[:limit]]
    return []


def _serialize_run(run):
    start_time = getattr(run, "start_time", None)
    end_time = getattr(run, "end_time", None)
    error = getattr(run, "error", None)
    total_tokens = getattr(run, "total_tokens", None)
    prompt_tokens = getattr(run, "prompt_tokens", None)
    completion_tokens = getattr(run, "completion_tokens", None)
    return {
        "id": _safe_id(getattr(run, "id", None)),
        "trace_id": _safe_id(getattr(run, "trace_id", None)),
        "parent_run_id": _safe_id(getattr(run, "parent_run_id", None)),
        "name": getattr(run, "name", None),
        "run_type": getattr(run, "run_type", None),
        "dotted_order": getattr(run, "dotted_order", None),
        "start_time": _to_iso(start_time),
        "end_time": _to_iso(end_time),
        "latency_ms": _latency_ms(start_time, end_time),
        "error": error,
        "status": "error" if error else ("completed" if end_time else "running"),
        "total_tokens": int(total_tokens) if isinstance(total_tokens, (int, float)) else None,
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None,
        "input_keys": _dict_keys(getattr(run, "inputs", None)),
        "output_keys": _dict_keys(getattr(run, "outputs", None)),
    }


def _build_trace_data(client, project_name, root_run):
    root_id = getattr(root_run, "id", None)
    trace_id = getattr(root_run, "trace_id", None) or root_id

    all_runs = []
    if trace_id:
        try:
            all_runs = list(client.list_runs(project_name=project_name, trace_id=trace_id, limit=99))
        except Exception:
            pass

    deduped = {}
    for run in [root_run, *all_runs]:
        run_id = _safe_id(getattr(run, "id", None))
        if run_id and run_id not in deduped:
            deduped[run_id] = run

    runs = list(deduped.values())
    runs.sort(key=lambda r: (_to_iso(getattr(r, "start_time", None)) or "", getattr(r, "dotted_order", "") or ""))

    serialized = [_serialize_run(r) for r in runs]
    error_count = sum(1 for r in serialized if r.get("error"))
    token_sum = sum(r.get("total_tokens") or 0 for r in serialized)

    root_serialized = _serialize_run(root_run)
    return {
        "project_name": project_name,
        "run_name": getattr(root_run, "name", None),
        "root_run_id": _safe_id(root_id),
        "trace_id": _safe_id(trace_id),
        "run_count": len(serialized),
        "error_count": error_count,
        "total_tokens": token_sum if token_sum > 0 else None,
        "root_latency_ms": root_serialized.get("latency_ms"),
        "runs": serialized,
    }


graph = None

# Try to import the compiled graph — it might be named 'graph' or 'agent'
try:
    from agent import graph
except ImportError:
    try:
        from agent import agent as graph
    except ImportError as e:
        print(json.dumps({"error": f"Failed to import 'graph' or 'agent' from agent.py: {e}"}))
        sys.exit(1)
except Exception as e:
    print(json.dumps({"error": f"Error loading agent: {e}"}))
    sys.exit(1)

# Read the input messages from a JSON file (written by sandbox_tools.py)
input_file = "/home/user/project/_chat_input.json"
try:
    with open(input_file, "r") as f:
        input_data = json.load(f)
    messages = input_data.get("messages", [])
except FileNotFoundError:
    print(json.dumps({"error": f"Input file not found: {input_file}"}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": f"Failed to parse input messages: {e}"}))
    sys.exit(1)

try:
    # Convert message dicts to tuples that LangGraph expects
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "human")
        content = msg.get("content", "")
        formatted_messages.append((role, content))

    # Invoke with an explicit run_name so we can reliably resolve this trace URL.
    final_state = graph.invoke(
        {"messages": formatted_messages},
        config={"run_name": run_name},
    )

    # The graph returns the updated state.
    # We want to extract the last AI message
    returned_messages = final_state.get("messages", [])
    if returned_messages:
        last_msg = returned_messages[-1]
        output_msg = {
            "role": last_msg.type if hasattr(last_msg, "type") else "assistant",
            "content": last_msg.content if hasattr(last_msg, "content") else str(last_msg),
        }
        # Prepare response
        response_data = {"message": output_msg}

        # Try to get LangSmith trace URL
        try:
            client = Client()
            project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"

            # Poll briefly because run indexing can be slightly delayed.
            trace_url = None
            trace_data = None
            for _ in range(6):
                recent_runs = list(client.list_runs(project_name=project_name, limit=25))
                matching_run = next((r for r in recent_runs if getattr(r, "name", "") == run_name), None)
                if matching_run:
                    trace_url = client.get_run_url(run=matching_run)
                    try:
                        trace_data = _build_trace_data(client, project_name, matching_run)
                    except Exception as e:
                        pass
                    break
                time.sleep(0.4)

            if trace_url:
                response_data["trace_url"] = trace_url
            if trace_data:
                response_data["trace_data"] = trace_data
        except Exception:
            # Don't fail the whole chat just because LangSmith URL extraction failed
            pass

        print(json.dumps(response_data))
    else:
        print(json.dumps({"error": "Graph returned no messages"}))

except Exception as e:
    import traceback
    tb = traceback.format_exc()
    print(json.dumps({"error": f"Graph execution failed: {e}\n\n--- Stack Trace ---\n{tb}"}))
    sys.exit(1)
