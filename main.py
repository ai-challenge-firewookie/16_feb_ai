import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from day2 import Day2Demo

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

client = Anthropic(api_key=api_key)
model = "claude-opus-4-6"
day2_demo = Day2Demo(client=client, model=model)

history: list[dict[str, str]] = []

print("Введите сообщение (пустая строка или /exit для выхода).")
while True:
    user_text = input("> ").strip()
    if not user_text or user_text == "/exit":
        break
    if user_text == "/day2":
        base_prompt = input("Базовый запрос для сравнения: ").strip()
        if not base_prompt:
            print("Пустой запрос. Возврат в чат.")
            continue
        wait_msg = "Waiting for response..."
        sys.stdout.write(f"\r{wait_msg}")
        sys.stdout.flush()
        pair = day2_demo.run(base_prompt=base_prompt)
        sys.stdout.write("\r" + " " * len(wait_msg) + "\r")
        sys.stdout.flush()
        print("\nБез ограничений:\n")
        print(pair.unconstrained)
        print("\nС ограничениями:\n")
        print(pair.constrained)
        continue

    history.append({"role": "user", "content": user_text})
    wait_msg = "Waiting for response..."
    sys.stdout.write(f"\r{wait_msg}")
    sys.stdout.flush()
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=history,
    )
    sys.stdout.write("\r" + " " * len(wait_msg) + "\r")
    sys.stdout.flush()
    assistant_text = message.content[0].text
    print(assistant_text)
    history.append({"role": "assistant", "content": assistant_text})
