from textwrap import dedent
from ollama import chat, Message
import argparse
from mcp_tools import get_weather, schedule_event
import json
from json import JSONDecodeError

TOOL_REGISTRY = {"get_weather": get_weather, "schedule_event": schedule_event}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature by city.",
            "parameters": {
                "type": "object",
                "properties": {"loc": {"type": "string", "description": "City name only, e.g., 'Carmel'"}},
                "required": ["loc"],
            },
        },
    }
]


def main(msg: str) -> None:
    default_system_msg = Message(
        role="system",
        # content=(
        #     'You are a helpful assistant. You can use tools by outputting a JSON object like {"tool": "tool_name", "args": {"param1": "value"}}'
        #     "Only do this if the query requires it; otherwise, respond directly."
        #     "Available tools: - get_weather: Fetches weather for a location. Args: {'location': 'city'}."
        #     "Answer in all lowercase letters."
        # ),
        content=dedent(
            """
        You are a helpful assistant. You can use tools by outputting a JSON object like {"tool": "tool_name", "args": {"param1": "value"}}
        Only do this if the query requires it; otherwise, respond directly.
        
        Available tools:
        - get_weather: Fetches weather for a location. Args: {'location': 'city'}.
        - schedule_event:  Schedules an event. Args: {"name": "event name", "time": "time"}.
        
        Answer in all lowercase letters.
        """
        ).strip(),
    )
    user_msg = Message(role="user", content=msg)

    messages: list[Message] = [default_system_msg, user_msg]

    while True:
        assistant_parts: list[str] = []

        stream_response = chat(
            # model="llama3.1:70b",
            model="dolphin3:8b-llama3.1-fp16",
            messages=messages,
            stream=True,
        )

        for chunk in stream_response:
            # Print assistant text.
            if chunk.message and chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                assistant_parts.append(chunk.message.content)

        print()  # Single newline after the final chunk for readability.

        # === Post streaming. ===
        assistant_txt = "".join(assistant_parts).strip()

        # === Always append assistant turn (content + tool calls) as a typed Message. ===
        assistant_msg = Message(role="assistant", content=assistant_txt)
        messages.append(assistant_msg)

        # === Try to parse a tool-call json. ===
        tool_call = None
        try:
            obj = json.loads(assistant_txt)
            if isinstance(obj, dict):
                tool_call = obj

        except JSONDecodeError:
            pass

        if not tool_call:
            break  # no tools requested => done.

        # === Execute tool and append result. ===
        tool_name = tool_call.get("tool")
        tool_params = tool_call.get("args")

        tool_func = TOOL_REGISTRY.get(tool_name)
        assert tool_func, f"Unknown tool: {tool_name}"

        try:
            tool_result = tool_func(**tool_params)
        except TypeError:
            # Fallback for positional args.
            tool_result = tool_func(tool_params.get("loc", ""))
        except Exception as e:
            tool_result = f"ERROR: tool {tool_name} failed: {e}"

        messages.append(Message(role="tool", tool_name=tool_name, content=str(tool_result)))

    print()
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LlamaBakery entry script.")

    p.add_argument("-m", "--message", required=True, help="Message to send to the LLM.")
    args = p.parse_args()

    main(msg=args.message)
