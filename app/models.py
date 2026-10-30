from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    email = Column(String(256), unique=True, nullable=True, index=True)
    password_hash = Column(String(256), nullable=True)

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String(256), nullable=False)
    content = Column(Text, nullable=False)
    source_url = Column(String(256), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="documents") 
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")  


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String(256), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")     


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")
    user = relationship("User", back_populates="chats")
