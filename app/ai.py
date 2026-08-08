import os

import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required for the chatbot.")

openai.api_key = OPENAI_API_KEY


def generate_response(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a production-style chatbot API."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
