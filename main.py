from typing import Sequence
from ollama import chat, Message
import argparse
from mcp_tools import get_weather

MODEL_TAG = "dolphin-l31-tools"

TOOL_REGISTRY = {
    "get_weather": get_weather,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature by city.",
            "parameters": {
                "type": "object",
                "properties": {"loc": {"type": "string", "description": "City, e.g., 'Carmel'"}},
                "required": ["loc"],
            },
        },
    }
]


def main(msg: str) -> None:
    default_system_msg = Message(
        role="system",
        content=(
            "You are my girlfriend texting me. Format your messages like a teenage text only. Not too formal, but not too informal. "
            "Tools are optional. Only call a tool if it is strictly necessary to answer accurately. "
            "If a tool is not needed, answer directly and do not mention tools or functions. "
        ),
    )
    user_msg = Message(role="user", content=msg)

    messages: list[Message] = [default_system_msg, user_msg]

    while True:
        last_tool_calls: Sequence[Message.ToolCall] | None = None
        assistant_parts: list[str] = []

        stream_response = chat(
            # model="llama3.1:70b",
            # model="dolphin3:8b-llama3.1-fp16",
            model=MODEL_TAG,
            messages=messages,
            tools=TOOLS_SCHEMA,
            stream=True,
        )

        for chunk in stream_response:
            # Print assistant text.
            if chunk.message and chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                assistant_parts.append(chunk.message.content)

            # Collect tool calls (may be multiple)
            if chunk.message and chunk.message.tool_calls:
                # Keep only the final, complete set of tool calls.
                last_tool_calls = chunk.message.tool_calls

        # === Post streaming ===
        assistant_txt = "".join(assistant_parts).strip()

        # Try to parse a tool-call json.
        # tool_call = None
        # try:
        #     obj = json.loads(assistant_txt)
        #     if isinstance(obj, dict):
        #         tool_call = obj
        #
        # except JSONDecodeError:
        #     pass

        # Always append assistant turn (content + tool calls) as a typed Message.
        assistant_msg = Message(role="assistant", content=assistant_txt, tool_calls=last_tool_calls)
        messages.append(assistant_msg)

        if not last_tool_calls:
            break  # no tools requested => done.

        # Execute tools and send results back.
        print(f"TOOL CALL => {last_tool_calls}")

        for call in last_tool_calls:
            name = call.function.name
            call_args = dict(call.function.arguments or {})
            func = TOOL_REGISTRY.get(name)

            if not func:
                messages.append(Message(role="tool", tool_name=name, content=f"ERROR: unknown tool'{name}'"))
                continue

            try:
                result = func(**call_args)
            except TypeError:
                result = func(**call_args) if call_args else func("")
            except Exception as e:
                result = f"ERROR: tool '{name}' failed: {e}"

            messages.append(Message(role="tool", tool_name=name, content=str(result)))

        # if not tool_call:
        #     print(f"DEBUG: No tool calls.")
        #     break  # no tools requested => done.
        #
        # # === Execute tool and append result ===
        # tool_name = tool_call.get("name")
        # tool_params = tool_call.get("parameters")
        # tool_func = TOOL_REGISTRY.get(tool_name)
        #
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
