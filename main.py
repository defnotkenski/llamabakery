from ollama import chat
import argparse


def main(msg: str | None) -> None:
    if msg is None:
        msg = "Why is the sky blue?"

    messages = [
        {
            "role": "user",
            "content": msg,
        }
    ]

    stream_response = chat(model="dolphin3:8b-llama3.1-fp16", messages=messages, stream=True)

    # print(response.message.content)
    for chunk in stream_response:
        print(chunk.message.content, end="", flush=True)

    print("")
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LlamaBakery entry script.")

    p.add_argument("-m", "--message", required=False, help="Message to send to the LLM.")
    args = p.parse_args()

    main(msg=args.message)
