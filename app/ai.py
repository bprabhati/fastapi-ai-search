import hashlib
import os

from openai import OpenAI

USE_MOCK_AI = os.getenv("USE_MOCK_AI", "false").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not USE_MOCK_AI and not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required for the chatbot.")

client = OpenAI(api_key=OPENAI_API_KEY) if not USE_MOCK_AI else None

EMBEDDING_DIMENSIONS = 1536


def generate_response(prompt: str) -> str:
    if USE_MOCK_AI:
        return f"[mock reply] You said: {prompt}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a production-style chatbot API."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()


def generate_embedding(text: str) -> list[float]:
    if USE_MOCK_AI:
        # deterministic pseudo-embedding so identical text always maps to the same vector
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        return [((seed >> i) % 1000) / 1000 for i in range(EMBEDDING_DIMENSIONS)]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks
