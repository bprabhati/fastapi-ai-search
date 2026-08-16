from typing import Optional

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    name: str
    email: Optional[EmailStr] = None


class UserRead(BaseModel):
    id: int
    name: str
    email: Optional[EmailStr] = None

    class Config:
        orm_mode = True


class ChatRequest(BaseModel):
    prompt: str
    user_id: Optional[int] = None


class ChatResponse(BaseModel):
    reply: str


class DocumentCreate(BaseModel):
    title: str
    content: str
    source_url: Optional[str] = None
    user_id: Optional[int] = None


class DocumentRead(BaseModel):
    id: int
    title: str
    source_url: Optional[str] = None
    chunk_count: int
