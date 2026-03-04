import json
import logging
import os
import sys

# Add the project dir to Python path so we can import the agent
sys.path.insert(0, "/home/user/project")

try:
    from agent import graph
except ImportError as e:
    print(json.dumps({"error": f"Failed to import 'graph' from agent: {e}"}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": f"Error loading agent: {e}"}))
    sys.exit(1)

# Read the input messages JSON from stdin
try:
    input_data = json.loads(sys.stdin.read())
    messages = input_data.get("messages", [])
except Exception as e:
    print(json.dumps({"error": f"Failed to parse input messages: {e}"}))
    sys.exit(1)

try:
    # Invoke the graph with the provided messages
    # Assuming the state has a 'messages' key
    final_state = graph.invoke({"messages": messages})

    # The graph returns the updated state.
    # We want to extract the last AI message
    returned_messages = final_state.get("messages", [])
    if returned_messages:
        # Get the last message, which should be the AI's response
        last_msg = returned_messages[-1]
        
        # Format for output
        output_msg = {
            "role": last_msg.type if hasattr(last_msg, "type") else "assistant",
            "content": last_msg.content,
        }
        
        # Include tool calls if present
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            output_msg["tool_calls"] = last_msg.tool_calls
            
        print(json.dumps({"message": output_msg}))
    else:
        print(json.dumps({"error": "Graph returned no messages"}))

except Exception as e:
    print(json.dumps({"error": f"Graph execution failed: {e}"}))
    sys.exit(1)
