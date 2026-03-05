"""Self-healing sandbox agent — orchestrates E2B sandbox with auto-debug loop.

Registered as 'sandbox_agent' in langgraph.json.

Graph flow:
  create_and_run action:
    route → create_sandbox → install_deps → run_agent → check_result
      ↳ error + retries left → debug_and_fix → install_deps (loop)
      ↳ error + no retries   → format_response (with error)
      ↳ success              → format_response

  chat action:
    route → handle_chat → check_chat → (fix_chat_error → handle_chat) | format_response
"""

import json
import logging
from typing import Annotated, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from src.prompts.sandbox_debug_prompt import sandbox_debug_prompt, sandbox_chat_debug_prompt
from src.tools.sandbox_tools import (
    create_sandbox_and_write_files,
    install_sandbox_deps,
    install_extra_packages,
    run_sandbox_agent,
    rewrite_sandbox_files,
    read_sandbox_files,
    chat_with_agent,
)

logger = logging.getLogger(__name__)

# Debug LLM — lazy-initialized to avoid import-time env var issues
_debug_model = None


def _get_debug_model():
    global _debug_model
    if _debug_model is None:
        _debug_model = init_chat_model("openai:gpt-4o-mini")
    return _debug_model

MAX_RETRIES = 3
MAX_CHAT_RETRIES = 2


# ═══════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════

from typing import TypedDict


class SandboxState(TypedDict):
    messages: list                # LangGraph input/output messages
    action: str                   # "create_and_run" | "chat"
    files: list[dict]             # [{filename, content}]
    sandbox_id: str | None
    steps: list[dict]             # Execution step log
    error: str | None
    retry_count: int
    max_retries: int
    # Chat-specific
    chat_messages: list[dict]
    chat_response: dict | None
    chat_retry_count: int


# Module-level registry for non-serializable E2B Sandbox objects
# Keyed by sandbox_id so nodes can look them up without storing in state
_sandbox_registry: dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════


def route_action(state: SandboxState) -> SandboxState:
    """Parse the input JSON and set action + fields in state."""
    messages = state.get("messages", [])

    # Find last human message
    human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            human_msg = msg
            break
        if isinstance(msg, dict) and msg.get("role") in ("human", "user"):
            human_msg = msg
            break

    if not human_msg:
        return {**state, "error": "No input message received", "action": "error"}

    content = human_msg.content if hasattr(human_msg, "content") else human_msg.get("content", "")

    try:
        action_data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {**state, "error": f"Invalid JSON input: {str(content)[:200]}", "action": "error"}

    action = action_data.get("action", "unknown")
    logger.info(f"Sandbox agent action: {action}")

    return {
        **state,
        "action": action,
        "files": action_data.get("files", state.get("files", [])),
        "sandbox_id": action_data.get("sandbox_id", state.get("sandbox_id")),
        "chat_messages": action_data.get("chat_messages", []),
        "steps": [],
        "error": None,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "chat_retry_count": 0,
    }


def create_sandbox(state: SandboxState) -> SandboxState:
    """Create E2B sandbox and write project files."""
    files = state.get("files", [])
    if not files:
        return {**state, "error": "No files provided"}

    result = create_sandbox_and_write_files(files)

    # Store sandbox object in registry (not in state — it's not serializable)
    if result["sandbox"] and result["sandbox_id"]:
        _sandbox_registry[result["sandbox_id"]] = result["sandbox"]

    return {
        **state,
        "sandbox_id": result["sandbox_id"],
        "steps": state.get("steps", []) + result["steps"],
        "error": result["error"],
    }


def rewrite_files(state: SandboxState) -> SandboxState:
    """Rewrite updated files to an existing sandbox."""
    sandbox = _sandbox_registry.get(state.get("sandbox_id"))
    if not sandbox:
        return {**state, "error": "No sandbox available for hot reload"}

    files = state.get("files", [])
    if not files:
        return {**state, "error": "No files provided"}

    result = rewrite_sandbox_files(sandbox, files)

    return {
        **state,
        "steps": state.get("steps", []) + result["steps"],
        "error": result["error"],
    }


def install_deps(state: SandboxState) -> SandboxState:
    """Install Python dependencies in the sandbox."""
    sandbox = _sandbox_registry.get(state.get("sandbox_id"))
    if not sandbox:
        return {**state, "error": "No sandbox available"}

    result = install_sandbox_deps(sandbox, state.get("files", []))

    return {
        **state,
        "steps": state.get("steps", []) + result["steps"],
        "error": result["error"],
    }


def run_agent(state: SandboxState) -> SandboxState:
    """Run the main agent file."""
    sandbox = _sandbox_registry.get(state.get("sandbox_id"))
    if not sandbox:
        return {**state, "error": "No sandbox available"}

    result = run_sandbox_agent(sandbox, state.get("files", []))

    return {
        **state,
        "steps": state.get("steps", []) + result["steps"],
        "error": result["error"],
    }


def debug_and_fix(state: SandboxState) -> SandboxState:
    """Use LLM to diagnose the error and produce fixed files."""
    sandbox = _sandbox_registry.get(state.get("sandbox_id"))
    error = state.get("error", "")
    files = state.get("files", [])
    retry_count = state.get("retry_count", 0)

    logger.info(f"Debug attempt {retry_count + 1}/{state.get('max_retries', MAX_RETRIES)}: {error[:200]}")

    # Read current file contents from sandbox
    filenames = [f["filename"] for f in files]
    if sandbox:
        current_contents = read_sandbox_files(sandbox, filenames)
    else:
        current_contents = {f["filename"]: f["content"] for f in files}

    # Build the debug prompt
    files_str = "\n\n".join(
        f"=== {fn} ===\n{content}" for fn, content in current_contents.items()
    )

    debug_input = f"""## Error Output
```
{error}
```

## Current Source Files
{files_str}
"""

    try:
        response = _get_debug_model().invoke([
            {"role": "system", "content": sandbox_debug_prompt},
            {"role": "user", "content": debug_input},
        ])

        # Parse the JSON response
        response_text = response.content.strip()
        # Try to extract JSON from markdown fences if present
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
        if json_match:
            response_text = json_match.group(1).strip()

        fix_data = json.loads(response_text)

        diagnosis = fix_data.get("diagnosis", "Unknown issue")
        fixed_files = fix_data.get("fixed_files", [])
        install_packages = fix_data.get("install_packages", [])

        logger.info(f"Debug diagnosis: {diagnosis}")
        logger.info(f"Fixed {len(fixed_files)} files, {len(install_packages)} extra packages")

        # Record the debug step
        debug_step = {
            "action": "debug_fix",
            "command": f"Auto-fix attempt {retry_count + 1}: {diagnosis}",
            "stdout": f"Fixed files: {', '.join(f['filename'] for f in fixed_files) or 'none'}\nExtra packages: {', '.join(install_packages) or 'none'}",
            "stderr": "",
            "exit_code": 0,
        }

        new_steps = state.get("steps", []) + [debug_step]

        # Apply fixed files to sandbox
        if fixed_files and sandbox:
            rewrite_result = rewrite_sandbox_files(sandbox, fixed_files)
            new_steps += rewrite_result["steps"]

            # Update our files list with the fixes
            files_copy = [dict(f) for f in files]
            for fix in fixed_files:
                for i, f in enumerate(files_copy):
                    if f["filename"] == fix["filename"]:
                        files_copy[i]["content"] = fix["content"]
                        break
                else:
                    files_copy.append(fix)

            files = files_copy

        # Install extra packages if needed
        if install_packages and sandbox:
            pkg_result = install_extra_packages(sandbox, install_packages)
            new_steps += pkg_result["steps"]

        return {
            **state,
            "files": files,
            "steps": new_steps,
            "error": None,  # Clear error so we retry
            "retry_count": retry_count + 1,
        }

    except Exception as e:
        logger.error(f"Debug LLM error: {e}")
        return {
            **state,
            "steps": state.get("steps", []) + [{
                "action": "debug_fix",
                "command": f"Auto-fix attempt {retry_count + 1} failed",
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
            }],
            "retry_count": retry_count + 1,
        }


def handle_chat(state: SandboxState) -> SandboxState:
    """Process a chat message to the running agent."""
    sandbox_id = state.get("sandbox_id")
    chat_messages = state.get("chat_messages", [])

    if not sandbox_id:
        return {**state, "error": "No sandbox_id provided", "chat_response": None}

    result = chat_with_agent(sandbox_id, chat_messages)

    if result.get("error"):
        return {
            **state,
            "error": result["error"],
            "chat_response": None,
        }

    return {
        **state,
        "error": None,
        "chat_response": result,
    }


def fix_chat_error(state: SandboxState) -> SandboxState:
    """Diagnose and fix a runtime error that occurred during chat."""
    error = state.get("error", "")
    sandbox_id = state.get("sandbox_id")
    chat_retry = state.get("chat_retry_count", 0)
    files = state.get("files", [])

    logger.info(f"Chat debug attempt {chat_retry + 1}/{MAX_CHAT_RETRIES}: {error[:200]}")

    try:
        from e2b import Sandbox as E2BSandbox
        sandbox = E2BSandbox.connect(sandbox_id)
    except Exception as e:
        logger.error(f"Cannot connect to sandbox for chat debug: {e}")
        return {**state, "chat_retry_count": chat_retry + 1}

    # Read current files
    filenames = [f["filename"] for f in files]
    current_contents = read_sandbox_files(sandbox, filenames)

    files_str = "\n\n".join(
        f"=== {fn} ===\n{content}" for fn, content in current_contents.items()
    )

    last_user_msg = ""
    for msg in reversed(state.get("chat_messages", [])):
        if msg.get("role") == "human":
            last_user_msg = msg.get("content", "")
            break

    debug_input = f"""## Runtime Error
```
{error}
```

## User Message That Triggered It
{last_user_msg}

## Agent Source Files
{files_str}
"""

    try:
        response = _get_debug_model().invoke([
            {"role": "system", "content": sandbox_chat_debug_prompt},
            {"role": "user", "content": debug_input},
        ])

        response_text = response.content.strip()
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
        if json_match:
            response_text = json_match.group(1).strip()

        fix_data = json.loads(response_text)
        fixed_files = fix_data.get("fixed_files", [])
        install_packages = fix_data.get("install_packages", [])
        diagnosis = fix_data.get("diagnosis", "Unknown")

        logger.info(f"Chat debug diagnosis: {diagnosis}")

        new_steps = state.get("steps", []) + [{
            "action": "chat_debug_fix",
            "command": f"Chat auto-fix {chat_retry + 1}: {diagnosis}",
            "stdout": f"Fixed: {', '.join(f['filename'] for f in fixed_files) or 'none'}",
            "stderr": "",
            "exit_code": 0,
        }]

        if fixed_files:
            rewrite_sandbox_files(sandbox, fixed_files)

            files_copy = [dict(f) for f in files]
            for fix in fixed_files:
                for i, f in enumerate(files_copy):
                    if f["filename"] == fix["filename"]:
                        files_copy[i]["content"] = fix["content"]
                        break
                else:
                    files_copy.append(fix)
            files = files_copy

        if install_packages:
            install_extra_packages(sandbox, install_packages)

        return {
            **state,
            "files": files,
            "steps": new_steps,
            "error": None,
            "chat_retry_count": chat_retry + 1,
        }

    except Exception as e:
        logger.error(f"Chat debug LLM error: {e}")
        return {**state, "chat_retry_count": chat_retry + 1}


def format_response(state: SandboxState) -> SandboxState:
    """Format the final response as an AIMessage."""
    action = state.get("action", "")

    if action == "chat":
        if state.get("chat_response") and state["chat_response"].get("message"):
            result = {
                "message": state["chat_response"]["message"],
                "error": None,
            }
        else:
            result = {"error": state.get("error", "Chat failed after retries")}
    elif action == "error":
        result = {"error": state.get("error", "Unknown error")}
    else:
        result = {
            "sandbox_id": state.get("sandbox_id"),
            "steps": state.get("steps", []),
            "error": state.get("error"),
            "retry_count": state.get("retry_count", 0),
        }

    return {
        **state,
        "messages": [AIMessage(content=json.dumps(result))],
    }


# ═══════════════════════════════════════════════════════════
# Conditional edges
# ═══════════════════════════════════════════════════════════


def route_by_action(state: SandboxState) -> str:
    """Route to the appropriate flow based on action type."""
    action = state.get("action", "")
    if action == "create_and_run":
        return "create_sandbox"
    elif action == "hot_reload":
        return "rewrite_files"
    elif action == "chat":
        return "handle_chat"
    else:
        return "format_response"


def check_create_result(state: SandboxState) -> str:
    """After sandbox creation, check if there was an error."""
    if state.get("error"):
        return "format_response"
    return "install_deps"


def check_deps_result(state: SandboxState) -> str:
    """After dep install, check if error → debug or proceed to run."""
    if state.get("error"):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", MAX_RETRIES)
        if retry_count < max_retries:
            return "debug_and_fix"
        return "format_response"
    return "run_agent"


def check_rewrite_result(state: SandboxState) -> str:
    """After rewriting files, check if error prior to running."""
    if state.get("error"):
        return "format_response"
    return "run_agent"


def check_run_result(state: SandboxState) -> str:
    """After running, check if error → debug or success."""
    if state.get("error"):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", MAX_RETRIES)
        if retry_count < max_retries:
            return "debug_and_fix"
        return "format_response"
    return "format_response"


def after_debug_fix(state: SandboxState) -> str:
    """After debug-fix, loop back to install_deps to retry."""
    return "install_deps"


def check_chat_result(state: SandboxState) -> str:
    """After chat, check if error → fix or return."""
    if state.get("error"):
        chat_retry = state.get("chat_retry_count", 0)
        if chat_retry < MAX_CHAT_RETRIES:
            return "fix_chat_error"
        return "format_response"
    return "format_response"


def after_chat_fix(state: SandboxState) -> str:
    """After chat fix, retry the chat."""
    return "handle_chat"


# ═══════════════════════════════════════════════════════════
# Build the graph
# ═══════════════════════════════════════════════════════════


graph_builder = StateGraph(SandboxState)

# Add all nodes
graph_builder.add_node("route_action", route_action)
graph_builder.add_node("create_sandbox", create_sandbox)
graph_builder.add_node("rewrite_files", rewrite_files)
graph_builder.add_node("install_deps", install_deps)
graph_builder.add_node("run_agent", run_agent)
graph_builder.add_node("debug_and_fix", debug_and_fix)
graph_builder.add_node("handle_chat", handle_chat)
graph_builder.add_node("fix_chat_error", fix_chat_error)
graph_builder.add_node("format_response", format_response)

# Entry point
graph_builder.add_edge(START, "route_action")

# Route based on action type
graph_builder.add_conditional_edges("route_action", route_by_action, {
    "create_sandbox": "create_sandbox",
    "rewrite_files": "rewrite_files",
    "handle_chat": "handle_chat",
    "format_response": "format_response",
})

# Create → check → install or error
graph_builder.add_conditional_edges("create_sandbox", check_create_result, {
    "install_deps": "install_deps",
    "format_response": "format_response",
})

# Rewrite → check → run or error
graph_builder.add_conditional_edges("rewrite_files", check_rewrite_result, {
    "run_agent": "run_agent",
    "format_response": "format_response",
})

# Install → check → run or debug or error
graph_builder.add_conditional_edges("install_deps", check_deps_result, {
    "run_agent": "run_agent",
    "debug_and_fix": "debug_and_fix",
    "format_response": "format_response",
})

# Run → check → success or debug
graph_builder.add_conditional_edges("run_agent", check_run_result, {
    "debug_and_fix": "debug_and_fix",
    "format_response": "format_response",
})

# Debug → loop back to install_deps
graph_builder.add_conditional_edges("debug_and_fix", after_debug_fix, {
    "install_deps": "install_deps",
})

# Chat → check → fix or return
graph_builder.add_conditional_edges("handle_chat", check_chat_result, {
    "fix_chat_error": "fix_chat_error",
    "format_response": "format_response",
})

# Chat fix → retry chat
graph_builder.add_conditional_edges("fix_chat_error", after_chat_fix, {
    "handle_chat": "handle_chat",
})

# End
graph_builder.add_edge("format_response", END)

sandbox_agent = graph_builder.compile()

logger.info("Sandbox agent loaded")
