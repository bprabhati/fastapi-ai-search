import asyncio
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi import File, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from jwt import InvalidTokenError
from pydantic import ValidationError
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
CHAT_TITLE_MAX_CHARS = 60
SOCKET_AUTH_TIMEOUT_SECONDS = 10
SOCKET_UNAUTHORIZED = 4401
NO_CONTEXT_REPLY = (
    "I couldn't find anything to answer from. Upload a document first, "
    "or choose a different one under “Search in”."
)


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


def _retrieve_context(
    db: Session,
    user_id: int,
    prompt: str,
    history: list[dict[str, str]],
    document_id: int | None,
) -> tuple[str | None, list[str]]:
    query_embedding = ai.generate_embedding(_retrieval_query(prompt, history))
    similar_chunks = crud.search_similar_chunks(
        db, query_embedding, user_id, document_id=document_id
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
    titles = [
        chunk.document.title
        for chunk in document_ordered_chunks
        if getattr(chunk, "document", None) is not None
    ]
    return context, list(dict.fromkeys(titles))


@app.post("/chat", response_model=schemas.ChatResponse)
def chat(
    request: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    history = crud.get_recent_chat_history(db, current_user.id)
    user_id = current_user.id
    try:
        context, _ = _retrieve_context(db, user_id, request.prompt, history, request.document_id)
    except ai.OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc

    # Don't keep a DB connection checked out during the model call.
    db.close()

    if context is None:
        reply = NO_CONTEXT_REPLY
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


def _get_owned_session(db: Session, session_id: int, user_id: int) -> models.ChatSession:
    session = crud.get_chat_session(db, session_id, user_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return session


@app.get("/sessions", response_model=list[schemas.ChatSessionRead])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return crud.list_chat_sessions(db, current_user.id)


@app.get("/sessions/{session_id}/messages", response_model=list[schemas.ChatMessageRead])
def list_session_messages(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _get_owned_session(db, session_id, current_user.id)
    return crud.list_session_messages(db, session_id)


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    crud.delete_chat_session(db, _get_owned_session(db, session_id, current_user.id))


def _socket_user_id(token: str) -> int | None:
    try:
        user_id = decode_access_token(token)
    except InvalidTokenError:
        return None
    db = SessionLocal()
    try:
        return user_id if crud.get_user(db, user_id) is not None else None
    finally:
        db.close()


async def _authenticate_socket(websocket: WebSocket) -> int | None:
    # Browsers can't set headers on a WebSocket, and a token in the URL would end up in
    # server logs, so the token is sent as the first message instead.
    try:
        raw_message = await asyncio.wait_for(websocket.receive_text(), SOCKET_AUTH_TIMEOUT_SECONDS)
        auth = schemas.SocketAuth.model_validate_json(raw_message)
    except (TimeoutError, ValidationError):
        return None
    return await asyncio.to_thread(_socket_user_id, auth.token)


def _prepare_socket_answer(
    user_id: int,
    message: schemas.SocketMessage,
) -> tuple[list[dict[str, str]], str | None, list[str]]:
    db = SessionLocal()
    try:
        history = []
        if message.session_id is not None:
            if crud.get_chat_session(db, message.session_id, user_id) is None:
                raise ValueError("That conversation no longer exists.")
            history = crud.get_recent_chat_history(db, user_id, session_id=message.session_id)
        context, sources = _retrieve_context(
            db, user_id, message.prompt, history, message.document_id
        )
        return history, context, sources
    finally:
        db.close()


def _save_socket_answer(user_id: int, message: schemas.SocketMessage, reply: str) -> int:
    db = SessionLocal()
    try:
        session_id = message.session_id
        if session_id is None:
            title = message.prompt[:CHAT_TITLE_MAX_CHARS]
            session_id = crud.create_chat_session(db, user_id, title).id
        crud.create_chat_message(db, user_id, message.prompt, reply, session_id=session_id)
        return session_id
    finally:
        db.close()


async def _answer_on_socket(
    websocket: WebSocket,
    user_id: int,
    message: schemas.SocketMessage,
) -> None:
    reply = ""
    stopped = False
    try:
        history, context, sources = await asyncio.to_thread(
            _prepare_socket_answer, user_id, message
        )
        if context is None:
            reply = NO_CONTEXT_REPLY
            await websocket.send_json({"type": "token", "text": reply})
        else:
            await websocket.send_json({"type": "sources", "titles": sources})
            tokens = ai.stream_response(message.prompt, context=context, history=history)
            # next() blocks while the model works, so it runs in a thread to keep the
            # server free. Stop cancels this wait straight away.
            while (token := await asyncio.to_thread(next, tokens, None)) is not None:
                reply += token
                await websocket.send_json({"type": "token", "text": token})
    except asyncio.CancelledError:
        stopped = True
    except (ValueError, RuntimeError, TimeoutError) as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        return

    session_id = message.session_id
    if reply:
        session_id = await asyncio.to_thread(_save_socket_answer, user_id, message, reply)
    await websocket.send_json(
        {"type": "stopped" if stopped else "done", "session_id": session_id}
    )


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    await websocket.accept()
    answer_task = None
    try:
        user_id = await _authenticate_socket(websocket)
        if user_id is None:
            await websocket.close(code=SOCKET_UNAUTHORIZED)
            return
        await websocket.send_json({"type": "ready"})

        while True:
            raw_message = await websocket.receive_text()
            try:
                message = schemas.SocketMessage.model_validate_json(raw_message)
            except ValidationError:
                await websocket.send_json(
                    {"type": "error", "message": "That message could not be read."}
                )
                continue

            answering = answer_task is not None and not answer_task.done()
            if message.type == "stop":
                if answering:
                    answer_task.cancel()
            elif answering:
                await websocket.send_json(
                    {"type": "error", "message": "Please wait for the current answer to finish."}
                )
            elif not message.prompt.strip():
                await websocket.send_json({"type": "error", "message": "Please type a question."})
            else:
                answer_task = asyncio.create_task(_answer_on_socket(websocket, user_id, message))
    except WebSocketDisconnect:
        pass
    finally:
        # The browser left, so stop the model work. Exceptions here only mean sending failed.
        if answer_task is not None:
            answer_task.cancel()
            await asyncio.gather(answer_task, return_exceptions=True)
