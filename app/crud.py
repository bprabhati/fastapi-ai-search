from sqlalchemy import case, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from . import ai, models, schemas
from .security import hash_password

DEFAULT_TOP_K = 3
HISTORY_MESSAGE_MAX_CHARS = 500


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_name(db: Session, name: str):
    return db.query(models.User).filter(models.User.name == name).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(
        name=user.name,
        email=user.email,
        password_hash=hash_password(user.password),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_chat_session(db: Session, user_id: int, title: str):
    session = models.ChatSession(user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_chat_session(db: Session, session_id: int, user_id: int):
    return (
        db.query(models.ChatSession)
        .filter(models.ChatSession.id == session_id, models.ChatSession.user_id == user_id)
        .first()
    )


def list_chat_sessions(db: Session, user_id: int):
    return (
        db.query(models.ChatSession)
        .filter(models.ChatSession.user_id == user_id)
        .order_by(models.ChatSession.id.desc())
        .all()
    )


def list_session_messages(db: Session, session_id: int):
    return (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.id)
        .all()
    )


def delete_chat_session(db: Session, session: models.ChatSession) -> None:
    db.delete(session)
    db.commit()


def create_chat_message(
    db: Session,
    user_id: int | None,
    prompt: str,
    response: str,
    session_id: int | None = None,
):
    chat = models.ChatMessage(
        user_id=user_id,
        prompt=prompt,
        response=response,
        session_id=session_id,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def get_recent_chat_history(
    db: Session,
    user_id: int,
    limit: int = 2,
    session_id: int | None = None,
) -> list[dict[str, str]]:
    query = db.query(models.ChatMessage).filter(models.ChatMessage.user_id == user_id)
    if session_id is not None:
        query = query.filter(models.ChatMessage.session_id == session_id)
    rows = query.order_by(models.ChatMessage.id.desc()).limit(limit).all()
    history = []
    for row in reversed(rows):
        history.extend(
            [
                {"role": "user", "content": row.prompt},
                {"role": "assistant", "content": row.response[:HISTORY_MESSAGE_MAX_CHARS]},
            ]
        )
    return history


def create_document_with_chunks(
    db: Session,
    title: str,
    content: str,
    source_url: str | None,
    user_id: int | None,
    chunks: list[str],
    embeddings: list[list[float]],
):
    document = models.Document(
        title=title,
        content=content,
        source_url=source_url,
        user_id=user_id,
        chunks=[
            models.DocumentChunk(chunk_index=index, content=chunk, embedding=embedding)
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ],
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def search_similar_chunks(
    db: Session,
    query_embedding: list[float],
    user_id: int,
    top_k: int = DEFAULT_TOP_K,
    document_id: int | None = None,
):
    matching_dimension_embedding = case(
        (
            func.vector_dims(models.DocumentChunk.embedding) == len(query_embedding),
            models.DocumentChunk.embedding,
        ),
        else_=None,
    )
    query = (
        db.query(models.DocumentChunk)
        .join(models.Document)
        .filter(models.Document.user_id == user_id)
        .filter(func.vector_dims(models.DocumentChunk.embedding) == len(query_embedding))
    )
    if document_id is not None:
        query = query.filter(models.Document.id == document_id)
    return (
        query
        .order_by(matching_dimension_embedding.cosine_distance(query_embedding))
        .limit(top_k)
        .all()
    )


def list_documents(db: Session, user_id: int):
    return (
        db.query(models.Document.id, models.Document.title)
        .filter(models.Document.user_id == user_id)
        .order_by(models.Document.id)
        .all()
    )


def list_document_titles(db: Session, user_id: int) -> list[str]:
    rows = (
        db.query(models.Document.title)
        .filter(models.Document.user_id == user_id)
        .order_by(models.Document.id)
        .all()
    )
    return [title for (title,) in rows]


def reembed_user_document_chunks(db: Session, user_id: int) -> int:
    chunks = (
        db.query(models.DocumentChunk.id, models.DocumentChunk.content)
        .join(models.Document)
        .filter(models.Document.user_id == user_id)
        .all()
    )
    try:
        embeddings = ai.generate_embeddings([content for _, content in chunks])
        for (chunk_id, _), embedding in zip(chunks, embeddings):
            (
                db.query(models.DocumentChunk)
                .filter(models.DocumentChunk.id == chunk_id)
                .update(
                    {models.DocumentChunk.embedding: embedding},
                    synchronize_session=False,
                )
            )
        db.commit()
    except (RuntimeError, SQLAlchemyError, ValueError):
        db.rollback()
        raise
    return len(chunks)
