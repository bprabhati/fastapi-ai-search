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
