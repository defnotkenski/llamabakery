from typing import Sequence
from ollama import chat, Message
import argparse
from mcp_tools import weather_penis

TOOL_REGISTRY = {
    "weather_penis": weather_penis,
}


def main(msg: str) -> None:
    default_system_msg = Message(
        role="system",
        content="You are my girlfriend texting me. Format your messages like a teenage text only. Not too formal, but not too informal.",
    )
    user_msg = Message(role="user", content=msg)

    messages: list[Message] = [default_system_msg, user_msg]

    while True:
        last_tool_calls: Sequence[Message.ToolCall] | None = None
        assistant_parts: list[str] = []

        stream_response = chat(
            # model="dolphin3:8b-llama3.1-fp16",
            model="llama4:17b-scout-16e-instruct-q4_K_M",
            messages=messages,
            tools=[weather_penis],
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

        # Append assistant turn (content + tool calls) as a typed Message.
        assistant_msg = Message(role="assistant", content="".join(assistant_parts), tool_calls=last_tool_calls)
        messages.append(assistant_msg)

        if not last_tool_calls:
            break  # no tools requested => done.

        # Execute tools and send results back.
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
                result = func(args if args else "")
            except Exception as e:
                result = f"ERROR: tool '{name}' failed: {e}"

            messages.append(Message(role="tool", tool_name=name, content=str(result)))

    print()
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LlamaBakery entry script.")

    p.add_argument("-m", "--message", required=True, help="Message to send to the LLM.")
    args = p.parse_args()

    main(msg=args.message)
