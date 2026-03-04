"""E2B Sandbox utilities — create sandboxes, write files, install deps, run commands."""

import logging
import os
from pathlib import Path

from e2b import Sandbox

logger = logging.getLogger(__name__)

# Sandbox timeout: 5 minutes (in seconds)
SANDBOX_TIMEOUT_SECONDS = 5 * 60


def _create_sandbox():
    """Create a sandbox in a version-compatible way."""
    timeout = SANDBOX_TIMEOUT_SECONDS
    # Newer E2B SDKs prefer Sandbox.create(...)
    create = getattr(Sandbox, "create", None)
    if callable(create):
        return create(timeout=timeout)
    # Older SDKs construct directly
    try:
        return Sandbox(timeout=timeout)
    except TypeError:
        sandbox = Sandbox()
        # Best-effort timeout setting if supported
        if hasattr(sandbox, "set_timeout"):
            try:
                sandbox.set_timeout(timeout)
            except Exception:
                pass
        return sandbox


def _read_local_env() -> str | None:
    """Read the project-root .env file so its keys can be forwarded to the sandbox."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        return env_path.read_text()
    return None


def _extract_deps_from_pyproject(files: list[dict]) -> list[str]:
    """Extract dependency names from a pyproject.toml file content.

    Parses the dependencies list directly from the TOML text
    so we can pip-install them without needing a valid pyproject layout.
    """
    import re

    for f in files:
        if f["filename"] != "pyproject.toml":
            continue
        content = f["content"]
        # Match: dependencies = [ "pkg1>=1.0", "pkg2", ... ]
        m = re.search(
            r'dependencies\s*=\s*\[(.*?)\]',
            content,
            re.DOTALL,
        )
        if not m:
            continue
        raw = m.group(1)
        # Extract quoted strings
        deps = re.findall(r'"([^"]+)"', raw)
        return deps
    return []


def create_and_run(files: list[dict]) -> dict:
    """Create an E2B sandbox, write files, install deps, and run the agent.

    Args:
        files: List of {filename, content} dicts.

    Returns:
        Dict with sandbox_id, steps (list of command results), and any error.
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        return {"error": "E2B_API_KEY not set. Add it to your .env file."}

    steps = []

    try:
        # 1. Create sandbox
        logger.info("Creating E2B sandbox...")
        sandbox = _create_sandbox()
        sandbox_id = sandbox.sandbox_id
        logger.info(f"Sandbox created: {sandbox_id}")

        # 2. Write all files into /home/user/project/
        project_dir = "/home/user/project"
        sandbox.commands.run(f"mkdir -p {project_dir}")

        for f in files:
            filepath = f"{project_dir}/{f['filename']}"
            # Ensure parent directory exists
            parent = "/".join(filepath.split("/")[:-1])
            sandbox.commands.run(f"mkdir -p {parent}")
            sandbox.files.write(filepath, f["content"])
            logger.info(f"Wrote: {filepath}")

        steps.append({
            "action": "write_files",
            "command": f"Wrote {len(files)} files to {project_dir}",
            "stdout": "\n".join(f"  → {f['filename']}" for f in files),
            "stderr": "",
            "exit_code": 0,
        })

        # 2b. Write the local .env so the sandbox agent has real API keys
        local_env = _read_local_env()
        if local_env:
            sandbox.files.write(f"{project_dir}/.env", local_env)
            logger.info("Forwarded local .env to sandbox")

        # 3. Install dependencies
        # Extract deps directly from pyproject.toml content (avoids setuptools issues)
        deps = _extract_deps_from_pyproject(files)
        has_requirements = any(f["filename"] == "requirements.txt" for f in files)

        # Always ensure essential LLM provider packages are included
        # (AI-generated pyproject.toml files often omit these)
        baseline = [
            "langchain-openai", "langchain-anthropic",
            "langchain", "langgraph", "langsmith", "python-dotenv",
        ]
        if deps:
            existing_names = {d.split(">=")[0].split("<=")[0].split("==")[0].strip() for d in deps}
            for pkg in baseline:
                if pkg not in existing_names:
                    deps.append(pkg)

        if deps:
            dep_str = " ".join(f'"{d}"' for d in deps)
            logger.info(f"Installing {len(deps)} dependencies...")
            result = sandbox.commands.run(
                f"cd {project_dir} && pip install {dep_str}",
                timeout=180,
            )
            steps.append({
                "action": "install_deps",
                "command": f"pip install ({len(deps)} packages)",
                "stdout": _truncate(result.stdout, 2000),
                "stderr": _truncate(result.stderr, 1000),
                "exit_code": result.exit_code,
            })
        elif has_requirements:
            logger.info("Installing dependencies from requirements.txt...")
            result = sandbox.commands.run(
                f"cd {project_dir} && pip install -r requirements.txt",
                timeout=120,
            )
            steps.append({
                "action": "install_deps",
                "command": "pip install -r requirements.txt",
                "stdout": _truncate(result.stdout, 2000),
                "stderr": _truncate(result.stderr, 1000),
                "exit_code": result.exit_code,
            })
        else:
            # Fallback: install common LangChain dependencies
            logger.info("No dependency file found, installing langchain + langgraph...")
            result = sandbox.commands.run(
                f"cd {project_dir} && pip install langchain langgraph langsmith python-dotenv langchain-openai",
                timeout=120,
            )
            steps.append({
                "action": "install_deps",
                "command": "pip install langchain langgraph langsmith python-dotenv langchain-openai",
                "stdout": _truncate(result.stdout, 2000),
                "stderr": _truncate(result.stderr, 1000),
                "exit_code": result.exit_code,
            })

        # 4. Run the agent
        # Try to find the main entry point
        main_file = _find_main_file(files)
        if main_file:
            logger.info(f"Running: python {main_file}")
            result = sandbox.commands.run(
                f"cd {project_dir} && python {main_file}",
                timeout=60,
            )
            steps.append({
                "action": "run_agent",
                "command": f"python {main_file}",
                "stdout": _truncate(result.stdout, 3000),
                "stderr": _truncate(result.stderr, 2000),
                "exit_code": result.exit_code,
            })
        else:
            # Just list files and show structure
            result = sandbox.commands.run(f"cd {project_dir} && find . -type f")
            steps.append({
                "action": "list_files",
                "command": "find . -type f (no main entry point found)",
                "stdout": result.stdout,
                "stderr": "",
                "exit_code": 0,
            })

        return {
            "sandbox_id": sandbox_id,
            "steps": steps,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Sandbox error: {e}")
        return {
            "sandbox_id": None,
            "steps": steps,
            "error": str(e),
        }


def _find_main_file(files: list[dict]) -> str | None:
    """Determine the main entry point file to run."""
    # Priority order for main file detection
    candidates = ["agent.py", "main.py", "app.py", "graph.py", "run.py"]
    filenames = {f["filename"] for f in files}

    for candidate in candidates:
        if candidate in filenames:
            return candidate

    # Fall back to any .py file
    py_files = [f["filename"] for f in files if f["filename"].endswith(".py")]
    if len(py_files) == 1:
        return py_files[0]

    return None


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with an indicator if too long."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} total chars)"


def chat_with_agent(sandbox_id: str, messages: list[dict]) -> dict:
    """Send a chat message to an already running agent in the E2B sandbox.

    Args:
        sandbox_id: The ID of the E2B sandbox.
        messages: The full list of conversation messages.
    """
    logger.info(f"Connecting to sandbox {sandbox_id} for chat...")
    try:
        # Import Sandbox dynamically if not already available in scope
        # (It is imported at module level, but we use _create_sandbox usually)
        from e2b import Sandbox

        sandbox = Sandbox.connect(sandbox_id)

        # Upload the runner script
        runner_path = Path(__file__).parent / "sandbox_runner_template.py"
        if not runner_path.exists():
            return {"error": "Runner template not found regionally"}

        runner_code = runner_path.read_text()
        sandbox.files.write("/home/user/project/runner.py", runner_code)

        # Execute runner.py, pass messages via stdin
        import json
        payload = json.dumps({"messages": messages})
        
        result = sandbox.commands.run(
            "cd /home/user/project && python runner.py",
            timeout=60,
            on_stdout=lambda x: logger.debug(f"[Agent STDOUT] {x}"),
            on_stderr=lambda x: logger.warning(f"[Agent STDERR] {x}")
        )

        if result.exit_code != 0:
            logger.error(f"Agent execution failed: {result.stderr}")
            return {"error": f"Agent execution failed: {result.stderr}"}

        # The runner.py outputs JSON on stdout containing either {"message": ...} or {"error": ...}
        import re
        output_text = result.stdout
        
        # Try to extract the JSON output
        try:
            # The runner script prints the result as the last line usually
            lines = [l.strip() for l in output_text.split("\n") if l.strip()]
            if not lines:
                return {"error": "No output from agent"}
                
            last_line = lines[-1]
            response = json.loads(last_line)
            
            if "error" in response:
                return {"error": response["error"]}
                
            return {"message": response.get("message")}
            
        except json.JSONDecodeError:
            return {"error": f"Failed to parse agent output: {output_text}"}

    except Exception as e:
        logger.error(f"Sandbox chat error: {e}")
        return {"error": str(e)}
