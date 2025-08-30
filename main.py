from ollama import chat
from ollama import ChatResponse


def main() -> None:
    messages = [
        {
            "role": "user",
            "content": "Why is the sky blue?",
        }
    ]

    response: ChatResponse = chat(model="gemma3", messages=messages)
    print(response.message.content)

    return


if __name__ == "__main__":
    main()
