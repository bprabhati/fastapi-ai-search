import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI

AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").lower()
# Chat and embeddings can use different providers, so a paid chat model can reuse local embeddings.
EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or AI_PROVIDER).lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "250"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
if not {AI_PROVIDER, EMBEDDING_PROVIDER} <= {"mock", "openai", "ollama"}:
    raise RuntimeError("AI_PROVIDER and EMBEDDING_PROVIDER must be 'openai', 'ollama' or 'mock'.")
if "openai" in {AI_PROVIDER, EMBEDDING_PROVIDER} and not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required for the chatbot.")

client = OpenAI(api_key=OPENAI_API_KEY) if "openai" in {AI_PROVIDER, EMBEDDING_PROVIDER} else None

EMBEDDING_DIMENSIONS = int(
    os.getenv("EMBEDDING_DIMENSIONS") or (768 if EMBEDDING_PROVIDER == "ollama" else 1536)
)
EMBEDDING_BATCH_SIZE = 16


class OllamaTimeoutError(TimeoutError):
    pass


@contextmanager
def _ollama_response(endpoint: str, payload: dict):
    request = Request(
        f"{OLLAMA_BASE_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            yield response
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Make sure Ollama is running."
        ) from exc
    except TimeoutError as exc:
        raise OllamaTimeoutError(
            "Ollama did not respond within 120 seconds. Check that Ollama is running "
            "and try again; a smaller model may respond faster."
        ) from exc


def _ollama_request(endpoint: str, payload: dict) -> dict:
    with _ollama_response(endpoint, payload) as response:
        return json.loads(response.read().decode("utf-8"))


def _chat_messages(
    prompt: str,
    context: str | None,
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    system_content = (
        "You answer questions about the user's uploaded documents. The retrieved document context is "
        "the sole source of truth for facts about those documents. Conversation history is only for "
        "resolving references such as 'the first project'; it may contain earlier mistakes and must "
        "never override the document context. Do not add facts that are not explicitly supported by "
        "the retrieved context. If the context does not identify the requested fact, say that you "
        "could not find it in the uploaded documents. Keep project names, employers, technologies, "
        "and their associations exactly as stated in the context. When asked about an ordinal such as "
        "'the first project', use document order, not retrieval order. Keep answers under 150 words."
    )
    # The context changes every question, so it goes last: Ollama can only reuse the
    # already-processed start of a prompt, and everything after the first change is reprocessed.
    user_content = prompt
    if context:
        user_content = (
            "Retrieved document excerpts (source labels include document order):\n"
            f"{context}\n\nQuestion: {prompt}"
        )

    return [
        {"role": "system", "content": system_content},
        *(history or []),
        {"role": "user", "content": user_content},
    ]


def _ollama_chat_payload(messages: list[dict[str, str]], stream: bool) -> dict:
    return {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.2, "num_predict": OLLAMA_MAX_TOKENS},
    }


def generate_response(
    prompt: str,
    context: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    if AI_PROVIDER == "mock":
        if context:
            return f"[mock reply] Using context: {context[:60]}... | You said: {prompt}"
        return f"[mock reply] You said: {prompt}"

    messages = _chat_messages(prompt, context, history)
    if AI_PROVIDER == "ollama":
        response = _ollama_request("/api/chat", _ollama_chat_payload(messages, stream=False))
        return response["message"]["content"].strip()

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def stream_response(
    prompt: str,
    context: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> Iterator[str]:
    if AI_PROVIDER == "mock":
        for word in f"[mock reply] You said: {prompt}".split():
            time.sleep(0.05)
            yield f"{word} "
        return

    messages = _chat_messages(prompt, context, history)
    if AI_PROVIDER == "ollama":
        with _ollama_response("/api/chat", _ollama_chat_payload(messages, stream=True)) as response:
            # Ollama sends one JSON object per line, each holding the next piece of the answer.
            for line in response:
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama failed while answering: {chunk['error']}")
                text = chunk.get("message", {}).get("content")
                if text:
                    yield text
        return

    stream = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content if chunk.choices else None
        if text:
            yield text


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_PROVIDER == "mock":
        # deterministic pseudo-embedding so identical text always maps to the same vector
        seeds = [int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) for text in texts]
        return [[((seed >> i) % 1000) / 1000 for i in range(EMBEDDING_DIMENSIONS)] for seed in seeds]

    if EMBEDDING_PROVIDER == "ollama":
        response = _ollama_request(
            "/api/embed",
            {"model": OLLAMA_EMBEDDING_MODEL, "input": texts, "keep_alive": OLLAMA_KEEP_ALIVE},
        )
        embeddings = response["embeddings"]
        if any(len(embedding) != EMBEDDING_DIMENSIONS for embedding in embeddings):
            raise RuntimeError(
                f"Ollama embedding model returned {len(embeddings[0])} dimensions; "
                f"expected {EMBEDDING_DIMENSIONS}."
            )
        return embeddings

    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        embeddings.extend(_embed_batch(texts[start:start + EMBEDDING_BATCH_SIZE]))
    return embeddings


def generate_embedding(text: str) -> list[float]:
    return generate_embeddings([text])[0]


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
