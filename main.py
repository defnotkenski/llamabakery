import json
from ollama import chat
import argparse
from mcp_tools import weather_penis

TOOL_REGISTRY = {
    "weather_penis": weather_penis,
}


def _parse_args(obj):
    # obj may be a dict or a JSON string; handle both.
    if isinstance(obj, (dict, list)):
        return obj

    try:
        return json.loads(obj or "{}")
    except Exception:
        return {}


def main(msg: str) -> None:
    messages = [{"role": "user", "content": msg}]

    while True:
        pending_tool_calls = []

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

            # Collect tool calls (may be multiple)
            if chunk.message and chunk.message.tool_calls:
                pending_tool_calls.append(chunk.message.tool_calls)

        print()

        if not pending_tool_calls:
            # No tool calls => assistant turn is complete; we're done.
            break

        # Execute tool calls and send results back as tool messages.
        for call in pending_tool_calls:
            fn = getattr(call, "function", None) or call.get("function", {})
            name = getattr(fn, "name", None) or fn.get("name")
            tool_args = getattr(fn, "arguments", None) or fn.get("arguments")

            func = TOOL_REGISTRY.get(name)
            if not func:
                # Surface unknown tool gracefully.
                messages.append({"role": "tool", "name": name or "unknown", "content": "ERROR: unknown tool"})
                continue

            kwargs = _parse_args(tool_args)
            # Call the tool and append the result.
            try:
                result = func(**kwargs) if isinstance(kwargs, dict) else func(kwargs)
            except TypeError:
                # Fallback if the tool expects a single positional arg
                result = func(kwargs if kwargs else "")

            messages.append({"role": "tool", "name": name, "content": str(result)})

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LlamaBakery entry script.")

    p.add_argument("-m", "--message", required=True, help="Message to send to the LLM.")
    args = p.parse_args()

    main(msg=args.message)
