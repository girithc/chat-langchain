"""E2B Sandbox utilities — composable functions for sandbox lifecycle.

Each function handles one step: create, write files, install deps, run, rewrite.
The sandbox_graph.py orchestrates these into a self-healing loop.
"""

import json
import logging
import os
import re
from pathlib import Path

from e2b import Sandbox

logger = logging.getLogger(__name__)

# Sandbox timeout: 5 minutes (in seconds)
SANDBOX_TIMEOUT_SECONDS = 5 * 60


# ═══════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════


def _create_sandbox():
    """Create a sandbox in a version-compatible way."""
    timeout = SANDBOX_TIMEOUT_SECONDS
    create = getattr(Sandbox, "create", None)
    if callable(create):
        return create(timeout=timeout)
    try:
        return Sandbox(timeout=timeout)
    except TypeError:
        sandbox = Sandbox()
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
    """Extract dependency names from a pyproject.toml file content."""
    for f in files:
        if f["filename"] != "pyproject.toml":
            continue
        content = f["content"]
        m = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if not m:
            continue
        raw = m.group(1)
        deps = re.findall(r'"([^"]+)"', raw)
        return deps
    return []


def _find_main_file(files: list[dict]) -> str | None:
    """Determine the main entry point file to run."""
    candidates = ["agent.py", "main.py", "app.py", "graph.py", "run.py"]
    filenames = {f["filename"] for f in files}
    for candidate in candidates:
        if candidate in filenames:
            return candidate
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


# ═══════════════════════════════════════════════════════════
# Composable sandbox operations (used by sandbox_graph.py)
# ═══════════════════════════════════════════════════════════


def create_sandbox_and_write_files(files: list[dict]) -> dict:
    """Create an E2B sandbox and write all project files into it.

    Returns dict with: sandbox_id, sandbox (object), steps, error
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        return {"sandbox_id": None, "sandbox": None, "steps": [],
                "error": "E2B_API_KEY not set. Add it to your .env file."}

    try:
        logger.info("Creating E2B sandbox...")
        sandbox = _create_sandbox()
        sandbox_id = sandbox.sandbox_id
        logger.info(f"Sandbox created: {sandbox_id}")

        project_dir = "/home/user/project"
        sandbox.commands.run(f"mkdir -p {project_dir}")

        for f in files:
            filepath = f"{project_dir}/{f['filename']}"
            parent = "/".join(filepath.split("/")[:-1])
            sandbox.commands.run(f"mkdir -p {parent}")
            sandbox.files.write(filepath, f["content"])
            logger.info(f"Wrote: {filepath}")

        steps = [{
            "action": "write_files",
            "command": f"Wrote {len(files)} files to {project_dir}",
            "stdout": "\n".join(f"  → {f['filename']}" for f in files),
            "stderr": "",
            "exit_code": 0,
        }]

        # Forward local .env for real API keys
        local_env = _read_local_env()
        if local_env:
            sandbox.files.write(f"{project_dir}/.env", local_env)
            logger.info("Forwarded local .env to sandbox")

        return {
            "sandbox_id": sandbox_id,
            "sandbox": sandbox,
            "steps": steps,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Sandbox creation error: {e}")
        return {"sandbox_id": None, "sandbox": None, "steps": [],
                "error": str(e)}


def install_sandbox_deps(sandbox, files: list[dict]) -> dict:
    """Install Python dependencies in the sandbox.

    Returns dict with: steps, error
    """
    project_dir = "/home/user/project"
    steps = []

    try:
        deps = _extract_deps_from_pyproject(files)
        has_requirements = any(f["filename"] == "requirements.txt" for f in files)

        # Always ensure essential packages
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
            if result.exit_code != 0:
                return {"steps": steps, "error": f"Dependency install failed: {_truncate(result.stderr, 500)}"}
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
            if result.exit_code != 0:
                return {"steps": steps, "error": f"Dependency install failed: {_truncate(result.stderr, 500)}"}
        else:
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
            if result.exit_code != 0:
                return {"steps": steps, "error": f"Dependency install failed: {_truncate(result.stderr, 500)}"}

        return {"steps": steps, "error": None}

    except Exception as e:
        logger.error(f"Dependency install error: {e}")
        return {"steps": steps, "error": str(e)}


def install_extra_packages(sandbox, packages: list[str]) -> dict:
    """Install additional packages in the sandbox (used during debug-fix cycles)."""
    if not packages:
        return {"steps": [], "error": None}

    project_dir = "/home/user/project"
    pkg_str = " ".join(f'"{p}"' for p in packages)
    logger.info(f"Installing extra packages: {packages}")

    try:
        result = sandbox.commands.run(
            f"cd {project_dir} && pip install {pkg_str}",
            timeout=120,
        )
        step = {
            "action": "install_fix_deps",
            "command": f"pip install {', '.join(packages)}",
            "stdout": _truncate(result.stdout, 1000),
            "stderr": _truncate(result.stderr, 500),
            "exit_code": result.exit_code,
        }
        return {"steps": [step], "error": None if result.exit_code == 0 else _truncate(result.stderr, 300)}
    except Exception as e:
        return {"steps": [], "error": str(e)}


def run_sandbox_agent(sandbox, files: list[dict]) -> dict:
    """Run the main agent file in the sandbox.

    Returns dict with: steps, error, stdout, stderr, exit_code
    """
    project_dir = "/home/user/project"
    main_file = _find_main_file(files)

    if not main_file:
        result = sandbox.commands.run(f"cd {project_dir} && find . -type f")
        return {
            "steps": [{
                "action": "list_files",
                "command": "find . -type f (no main entry point found)",
                "stdout": result.stdout,
                "stderr": "",
                "exit_code": 0,
            }],
            "error": "No main entry point found (need agent.py, main.py, app.py, etc.)",
            "stdout": result.stdout,
            "stderr": "",
            "exit_code": 1,
        }

    try:
        logger.info(f"Running: python {main_file}")
        result = sandbox.commands.run(
            f"cd {project_dir} && python {main_file}",
            timeout=60,
        )
        step = {
            "action": "run_agent",
            "command": f"python {main_file}",
            "stdout": _truncate(result.stdout, 3000),
            "stderr": _truncate(result.stderr, 2000),
            "exit_code": result.exit_code,
        }
        error = None
        if result.exit_code != 0:
            error = f"Exit code {result.exit_code}: {_truncate(result.stderr or result.stdout, 1000)}"

        return {
            "steps": [step],
            "error": error,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "exit_code": result.exit_code,
        }

    except Exception as e:
        logger.error(f"Run agent error: {e}")
        return {
            "steps": [{
                "action": "run_agent",
                "command": f"python {main_file}",
                "stdout": "",
                "stderr": str(e),
                "exit_code": 1,
            }],
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
            "exit_code": 1,
        }


def rewrite_sandbox_files(sandbox, fixed_files: list[dict]) -> dict:
    """Write corrected files to an existing sandbox.

    Args:
        sandbox: E2B Sandbox object
        fixed_files: List of {filename, content} dicts (only changed files)

    Returns dict with: steps, error
    """
    project_dir = "/home/user/project"
    steps = []

    try:
        for f in fixed_files:
            filepath = f"{project_dir}/{f['filename']}"
            parent = "/".join(filepath.split("/")[:-1])
            sandbox.commands.run(f"mkdir -p {parent}")
            sandbox.files.write(filepath, f["content"])
            logger.info(f"Rewrote: {filepath}")

        steps.append({
            "action": "rewrite_files",
            "command": f"Updated {len(fixed_files)} files",
            "stdout": "\n".join(f"  ✏ {f['filename']}" for f in fixed_files),
            "stderr": "",
            "exit_code": 0,
        })

        return {"steps": steps, "error": None}

    except Exception as e:
        logger.error(f"File rewrite error: {e}")
        return {"steps": steps, "error": str(e)}


def read_sandbox_files(sandbox, filenames: list[str]) -> dict[str, str]:
    """Read current file contents from the sandbox (for sending to debug LLM)."""
    project_dir = "/home/user/project"
    contents = {}
    for fn in filenames:
        try:
            content = sandbox.files.read(f"{project_dir}/{fn}")
            contents[fn] = content
        except Exception:
            contents[fn] = "(file not found or unreadable)"
    return contents


def chat_with_agent(sandbox_id: str, messages: list[dict]) -> dict:
    """Send a chat message to an already running agent in the E2B sandbox.

    Args:
        sandbox_id: The ID of the E2B sandbox.
        messages: The full list of conversation messages.

    Returns dict with: message, error, stderr (for debug if error)
    """
    logger.info(f"Connecting to sandbox {sandbox_id} for chat...")
    try:
        sandbox = Sandbox.connect(sandbox_id)

        # Upload the runner script
        runner_path = Path(__file__).parent / "sandbox_runner_template.py"
        if not runner_path.exists():
            return {"error": "Runner template not found", "stderr": ""}

        runner_code = runner_path.read_text()
        sandbox.files.write("/home/user/project/runner.py", runner_code)

        # Write messages to a JSON file that runner.py will read
        payload = json.dumps({"messages": messages})
        sandbox.files.write("/home/user/project/_chat_input.json", payload)

        # Execute runner.py
        result = sandbox.commands.run(
            "cd /home/user/project && python runner.py",
            timeout=60,
        )

        if result.exit_code != 0:
            error_detail = result.stderr or result.stdout or "Unknown error"
            logger.error(f"Agent execution failed (exit {result.exit_code}): {error_detail}")
            return {
                "error": f"Command exited with code {result.exit_code}: {_truncate(error_detail, 500)}",
                "stderr": error_detail,
            }

        # Parse runner.py JSON output
        output_text = result.stdout
        try:
            lines = [l.strip() for l in output_text.split("\n") if l.strip()]
            if not lines:
                return {"error": "No output from agent", "stderr": ""}

            for line in reversed(lines):
                try:
                    response = json.loads(line)
                    if "error" in response:
                        return {"error": response["error"], "stderr": ""}
                    return {"message": response.get("message"), "error": None, "stderr": ""}
                except json.JSONDecodeError:
                    continue

            return {"error": f"No valid JSON in output: {output_text[:500]}", "stderr": ""}

        except Exception as e:
            return {"error": f"Failed to parse agent output: {e}", "stderr": ""}

    except Exception as e:
        logger.error(f"Sandbox chat error: {e}")
        return {"error": str(e), "stderr": str(e)}
