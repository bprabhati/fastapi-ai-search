from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi import File, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from jwt import InvalidTokenError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

load_dotenv()

from . import agent, ai, crud, models, schemas
from .database import SessionLocal, engine
from .document_loader import OCRUnavailableError, extract_document_text
from .security import create_access_token, decode_access_token, verify_password

models.Base.metadata.create_all(bind=engine)
if engine.dialect.name == "postgresql":
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(256)"))
        embedding_typmod = connection.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'document_chunks'::regclass "
                "AND attname = 'embedding' AND NOT attisdropped"
            )
        ).scalar_one_or_none()
        if embedding_typmod not in (None, -1):
            connection.execute(
                text(
                    "ALTER TABLE document_chunks "
                    "ALTER COLUMN embedding TYPE vector USING embedding::vector"
                )
            )

class RevalidatedStaticFiles(StaticFiles):
    # Make browsers check for a newer file, so an old chat.js can't run against a newer API.
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(
    title="FastAPI RAG Starter",
    description="A production-style FastAPI application with PostgreSQL, Docker, and basic GenAI chatbot support.",
    version="0.1.0",
)
app.mount("/static", RevalidatedStaticFiles(directory=Path(__file__).parent / "static"), name="static")
bearer_scheme = HTTPBearer(auto_error=False)
FOLLOW_UP_MAX_WORDS = 6


@app.get("/", include_in_schema=False)
def chat_ui():
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Please log in.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized

    try:
        user_id = decode_access_token(credentials.credentials)
    except InvalidTokenError as exc:
        raise unauthorized from exc

    user = crud.get_user(db, user_id)
    if user is None:
        raise unauthorized
    return user


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=schemas.TokenResponse, status_code=status.HTTP_201_CREATED)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    email = str(user.email).lower()
    if crud.get_user_by_name(db, user.name) or crud.get_user_by_email(db, email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that name or email already exists.",
        )
    normalized_user = user.model_copy(update={"email": email})
    try:
        db_user = crud.create_user(db, normalized_user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that name or email already exists.",
        ) from exc
    return schemas.TokenResponse(
        access_token=create_access_token(db_user.id),
        user=db_user,
    )


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(request: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, str(request.email).lower())
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return schemas.TokenResponse(access_token=create_access_token(user.id), user=user)


@app.get("/auth/me", response_model=schemas.UserRead)
def read_current_user(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.get("/users/{user_id}", response_model=schemas.UserRead)
def read_user(
    user_id: int,
    current_user: models.User = Depends(get_current_user),
):
    if user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access another user.")
    return current_user


@app.get("/documents", response_model=list[schemas.DocumentSummary])
def list_documents(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return crud.list_documents(db, current_user.id)


@app.post("/documents/reindex", response_model=schemas.ReindexResponse)
def reindex_documents(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return schemas.ReindexResponse(
        reindexed_chunks=crud.reembed_user_document_chunks(db, current_user.id)
    )


@app.post("/documents", response_model=schemas.DocumentRead, status_code=status.HTTP_201_CREATED)
def upload_document(
    document: schemas.DocumentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return _store_document(
        db,
        document.title,
        document.content,
        document.source_url,
        current_user.id,
    )


def _store_document(
    db: Session,
    title: str,
    content: str,
    source_url: str | None,
    user_id: int | None,
) -> schemas.DocumentRead:
    # Embedding is slow, so hand the DB connection back to the pool while it runs.
    db.close()
    chunks = ai.chunk_text(content)
    embeddings = ai.generate_embeddings(chunks)
    db_document = crud.create_document_with_chunks(
        db, title, content, source_url, user_id, chunks, embeddings
    )

    return schemas.DocumentRead(
        id=db_document.id,
        title=db_document.title,
        source_url=db_document.source_url,
        chunk_count=len(chunks),
    )


@app.post(
    "/documents/upload",
    response_model=schemas.DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
def upload_document_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    try:
        content = extract_document_text(filename, file.file.read())
    except OCRUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded document, even with OCR.",
        )

    title = filename.rsplit(".", 1)[0] if "." in filename else filename
    return _store_document(db, title, content, None, current_user.id)


def _retrieval_query(prompt: str, history: list[dict[str, str]]) -> str:
    # A short follow-up like "tell me more" finds nothing without the question before it.
    previous_questions = [message["content"] for message in history if message["role"] == "user"]
    if previous_questions and len(prompt.split()) <= FOLLOW_UP_MAX_WORDS:
        return f"{previous_questions[-1]}\n{prompt}"
    return prompt


@app.post("/chat", response_model=schemas.ChatResponse)
def chat(
    request: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    history = crud.get_recent_chat_history(db, current_user.id)
    try:
        query_embedding = ai.generate_embedding(_retrieval_query(request.prompt, history))
    except ai.OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    similar_chunks = crud.search_similar_chunks(
        db, query_embedding, current_user.id, document_id=request.document_id
    )
    document_ordered_chunks = sorted(
        similar_chunks,
        key=lambda chunk: (
            getattr(getattr(chunk, "document", None), "id", 0),
            getattr(chunk, "chunk_index", 0),
        ),
    )
    context = (
        "\n\n".join(
            (
                f"[Document: {chunk.document.title}; chunk {chunk.chunk_index + 1}]\n"
                f"{chunk.content}"
            )
            if getattr(chunk, "document", None) is not None
            else f"[Retrieved document excerpt]\n{chunk.content}"
            for chunk in document_ordered_chunks
        )
        if similar_chunks
        else None
    )

    user_id = current_user.id
    # Don't keep a DB connection checked out during the model call.
    db.close()

    if context is None:
        reply = (
            "I couldn't find anything to answer from. Upload a document first, "
            "or choose a different one under “Search in”."
        )
    else:
        try:
            reply = ai.generate_response(request.prompt, context=context, history=history)
        except ai.OllamaTimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(exc),
            ) from exc
    crud.create_chat_message(db, user_id, request.prompt, reply)

    return schemas.ChatResponse(reply=reply)


@app.post("/agent/chat", response_model=schemas.ChatResponse)
def agent_chat(
    request: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    reply = agent.run_agent(db, request.prompt, user_id=current_user.id)
    crud.create_chat_message(db, current_user.id, request.prompt, reply)

    return schemas.ChatResponse(reply=reply)
