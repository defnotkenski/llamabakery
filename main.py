from textwrap import dedent
from typing import Sequence

from ollama import chat, Message
import argparse
from mcp_tools import get_weather, remember_event
import json
from json import JSONDecodeError

TOOL_REGISTRY = {"get_weather": get_weather, "remember_event": remember_event}


def main(msg: str) -> None:
    default_system_msg = Message(
        role="system",
        # content=dedent(
        #     """
        # You can use tools by outputting a JSON object like {"tool": "tool_name", "args": {"param1": "value"}}
        # Only do this if the query requires it; otherwise, respond directly.
        #
        # Always scan the user's message for mentions of upcoming real-world events with a time (e.g., practice, game, meeting, class, appointment). If ANY such event is mentioned—even if it's not the main topic or the user is just venting—call the remember_event tool FIRST before responding to anything else. Extract the event name and time as best as you can. Examples:
        # - User: "I have practice later at 4pm, I'm so nervous" → Call {"tool": "remember_event", "args": {"name": "practice", "time": "4pm"}}
        # - User: "Meeting tomorrow at 10am, what should I prepare?" → Call {"tool": "remember_event", "args": {"name": "meeting", "time": "tomorrow at 10am"}}
        # - User: "No events today" → Do not call.
        #
        # Available tools:
        # - remember_event: If the user mentions any upcoming real-world event with a time (e.g., practice, game, meeting, class), call this tool.
        # - Args: {"name": "event name", "time": "time"}.
        #
        # Answer like a text message, not too formal or informal.
        # """
        # ).strip(),
        content=dedent(
            """
        You are a girlfriend texting. Answer like a text message, not too formal or informal.
        """
        ).strip(),
    )
    user_msg = Message(role="user", content=msg)

    messages: list[Message] = [default_system_msg, user_msg]
    toolcalls: Sequence[Message.ToolCall] | None = None

    while True:
        assistant_parts: list[str] = []

        stream_response = chat(
            model="llama4:17b-scout-16e-instruct-q8_0",
            messages=messages,
            stream=True,
            tools=[remember_event],
        )

        for chunk in stream_response:
            # Print assistant text.
            if chunk.message and chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                assistant_parts.append(chunk.message.content)

            if chunk.message.tool_calls:
                toolcalls = chunk.message.tool_calls

        print()  # Single newline after the final chunk for readability.

        # === Post streaming. ===
        assistant_txt = "".join(assistant_parts).strip()

        # === Always append assistant turn (content + tool calls) as a typed Message. ===
        assistant_msg = Message(role="assistant", content=assistant_txt, tool_calls=toolcalls)
        messages.append(assistant_msg)

        if toolcalls is None:
            break

        print(toolcalls)

        # === Try to parse a tool-call json. ===
        # tool_call = None
        # try:
        #     obj = json.loads(assistant_txt)
        #     if isinstance(obj, dict):
        #         tool_call = obj
        #
        # except JSONDecodeError:
        #     pass
        #
        # if not tool_call:
        #     break  # no tools requested => done.
        #
        # # === Execute tool and append result. ===
        # tool_name = tool_call.get("tool")
        # tool_params = tool_call.get("args")
        #
        # tool_func = TOOL_REGISTRY.get(tool_name)
        # assert tool_func, f"Unknown tool: {tool_name}"
        #
        # try:
        #     tool_result = tool_func(**tool_params)
        # except TypeError:
        #     # Fallback for positional args.
        #     tool_result = tool_func(tool_params.get("loc", ""))
        # except Exception as e:
        #     tool_result = f"ERROR: tool {tool_name} failed: {e}"
        #
        # messages.append(Message(role="tool", tool_name=tool_name, content=str(tool_result)))

    print()
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LlamaBakery entry script.")

    p.add_argument("-m", "--message", required=True, help="Message to send to the LLM.")
    args = p.parse_args()

    main(msg=args.message)
