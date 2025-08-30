from ollama import chat


def main() -> None:
    messages = [
        {
            "role": "user",
            "content": "Why is the sky blue?",
        }
    ]

    stream_response = chat(model="gemma3", messages=messages, stream=True)

    # print(response.message.content)
    for chunk in stream_response:
        print(chunk.message.content, end="", flush=True)

    return


if __name__ == "__main__":
    main()
