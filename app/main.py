import os
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from . import ai, crud, models, schemas
from .database import SessionLocal, engine

load_dotenv()
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FastAPI RAG Starter",
    description="A production-style FastAPI application with PostgreSQL, Docker, and basic GenAI chatbot support.",
    version="0.1.0",
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/users", response_model=schemas.UserRead, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing_user = crud.get_user_by_name(db, user.name)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with name '{user.name}' already exists.",
        )
    return crud.create_user(db, user)


@app.get("/users/{user_id}", response_model=schemas.UserRead)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user


@app.post("/documents", response_model=schemas.DocumentRead, status_code=status.HTTP_201_CREATED)
def upload_document(document: schemas.DocumentCreate, db: Session = Depends(get_db)):
    db_document = crud.create_document(db, document.title, document.content, document.source_url, document.user_id)

    chunks = ai.chunk_text(document.content)
    for index, chunk_content in enumerate(chunks):
        embedding = ai.generate_embedding(chunk_content)
        crud.create_document_chunk(db, db_document.id, index, chunk_content, embedding)

    return schemas.DocumentRead(
        id=db_document.id,
        title=db_document.title,
        source_url=db_document.source_url,
        chunk_count=len(chunks),
    )


@app.post("/chat", response_model=schemas.ChatResponse)
def chat(request: schemas.ChatRequest, db: Session = Depends(get_db)):
    if request.user_id is not None:
        user = crud.get_user(db, request.user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    else:
        user = None

    reply = ai.generate_response(request.prompt)
    crud.create_chat_message(db, user.id if user else None, request.prompt, reply)

    return schemas.ChatResponse(reply=reply)
