"""A minimal tool-calling agent built on top of the existing RAG pieces.

Unlike /chat (which always does one embedding search then one LLM call), this
lets the model decide *if* and *how many times* to call tools before answering,
which is the core difference between a plain RAG pipeline and an "agent".
"""
import json

from sqlalchemy.orm import Session

from . import ai, crud

MAX_TOOL_TURNS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search stored documents for chunks relevant to a query. "
                "Use this whenever answering requires information from the user's documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "top_k": {"type": "integer", "description": "Number of chunks to return.", "default": crud.DEFAULT_TOP_K},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the titles of documents currently stored in the knowledge base.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _search_knowledge_base(db: Session, user_id: int, query: str, top_k: int = crud.DEFAULT_TOP_K) -> str:
    embedding = ai.generate_embedding(query)
    chunks = crud.search_similar_chunks(db, embedding, user_id=user_id, top_k=top_k)
    if not chunks:
        return "No relevant document chunks found."
    return "\n\n".join(chunk.content for chunk in chunks)


def _list_documents(db: Session, user_id: int) -> str:
    titles = crud.list_document_titles(db, user_id)
    return "\n".join(titles) if titles else "No documents stored yet."


def _execute_tool(db: Session, user_id: int, name: str, arguments: dict) -> str:
    if name == "search_knowledge_base":
        return _search_knowledge_base(db, user_id, arguments.get("query", ""), arguments.get("top_k", crud.DEFAULT_TOP_K))
    if name == "list_documents":
        return _list_documents(db, user_id)
    return f"Unknown tool: {name}"


def run_agent(db: Session, prompt: str, user_id: int) -> str:
    if ai.AI_PROVIDER != "openai":
        # Local Ollama and mock modes use the existing plain RAG path.
        context = _search_knowledge_base(db, user_id, prompt)
        return ai.generate_response(prompt, context=context)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant with access to tools for searching a document knowledge base. "
                "Call a tool when you need information you don't already have, then answer the user's question."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for _ in range(MAX_TOOL_TURNS):
        response = ai.client.chat.completions.create(
            model=ai.OPENAI_CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=400,
            temperature=0.7,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return (message.content or "").strip()

        messages.append(message.model_dump(exclude_unset=True))
        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = _execute_tool(db, user_id, tool_call.function.name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return "I couldn't complete the request within the allowed number of tool calls."
