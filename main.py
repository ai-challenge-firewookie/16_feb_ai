import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

client = Anthropic(api_key=api_key)

history: list[dict[str, str]] = []

print("Введите сообщение (пустая строка или /exit для выхода).")
while True:
    user_text = input("> ").strip()
    if not user_text or user_text == "/exit":
        break

    history.append({"role": "user", "content": user_text})
    wait_msg = "Waiting for response..."
    sys.stdout.write(f"\r{wait_msg}")
    sys.stdout.flush()
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=history,
    )
    sys.stdout.write("\r" + " " * len(wait_msg) + "\r")
    sys.stdout.flush()
    assistant_text = message.content[0].text
    print(assistant_text)
    history.append({"role": "assistant", "content": assistant_text})
