from ollama import chat
import argparse
from mcp_tools import weather_penis


def main(msg: str) -> None:
    messages = [
        {
            "role": "user",
            "content": msg,
        }
    ]

    stream_response = chat(
        model="dolphin3:8b-llama3.1-fp16",
        messages=messages,
        tools=[weather_penis],
        stream=True,
    )

    for chunk in stream_response:
        # Print content.
        print(chunk.message.content, end="", flush=True)

        # Print tool calls.
        if chunk.message.tool_calls:
            print(chunk.message.tool_calls)

    print()
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LlamaBakery entry script.")

    p.add_argument("-m", "--message", required=True, help="Message to send to the LLM.")
    args = p.parse_args()

    main(msg=args.message)
