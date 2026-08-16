from sqlalchemy.orm import Session

from . import models, schemas


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_name(db: Session, name: str):
    return db.query(models.User).filter(models.User.name == name).first()


def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_chat_message(db: Session, user_id: int | None, prompt: str, response: str):
    chat = models.ChatMessage(user_id=user_id, prompt=prompt, response=response)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def create_document(db: Session, title: str, content: str, source_url: str | None, user_id: int | None):
    document = models.Document(title=title, content=content, source_url=source_url, user_id=user_id)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def create_document_chunk(db: Session, document_id: int, chunk_index: int, content: str, embedding: list[float]):
    chunk = models.DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding,
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk
