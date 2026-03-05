import json
import logging
import os
import sys

# Add the project dir to Python path so we can import the agent
sys.path.insert(0, "/home/user/project")

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

    # Invoke the graph with the provided messages
    final_state = graph.invoke({"messages": formatted_messages})

    # The graph returns the updated state.
    # We want to extract the last AI message
    returned_messages = final_state.get("messages", [])
    if returned_messages:
        last_msg = returned_messages[-1]

        output_msg = {
            "role": last_msg.type if hasattr(last_msg, "type") else "assistant",
            "content": last_msg.content if hasattr(last_msg, "content") else str(last_msg),
        }

        print(json.dumps({"message": output_msg}))
    else:
        print(json.dumps({"error": "Graph returned no messages"}))

except Exception as e:
    print(json.dumps({"error": f"Graph execution failed: {e}"}))
    sys.exit(1)

