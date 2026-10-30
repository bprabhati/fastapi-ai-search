from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserRead(BaseModel):
    id: int
    name: str
    email: Optional[EmailStr] = None

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class ChatRequest(BaseModel):
    prompt: str
    document_id: Optional[int] = None


class ChatResponse(BaseModel):
    reply: str


class ChatSessionRead(BaseModel):
    id: int
    title: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ChatMessageRead(BaseModel):
    id: int
    prompt: str
    response: str

    model_config = ConfigDict(from_attributes=True)


class SocketAuth(BaseModel):
    type: Literal["auth"]
    token: str


class SocketMessage(BaseModel):
    type: Literal["question", "stop"]
    prompt: str = Field(default="", max_length=4000)
    document_id: Optional[int] = None
    session_id: Optional[int] = None


class DocumentCreate(BaseModel):
    title: str
    content: str
    source_url: Optional[str] = None


class DocumentSummary(BaseModel):
    id: int
    title: str

    model_config = ConfigDict(from_attributes=True)


class DocumentRead(BaseModel):
    id: int
    title: str
    source_url: Optional[str] = None
    chunk_count: int


class ReindexResponse(BaseModel):
    reindexed_chunks: int
