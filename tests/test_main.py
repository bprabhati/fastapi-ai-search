import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient
import pytest

from app.main import app, get_db
import app.ai as ai
import app.crud as crud

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_user_success(monkeypatch):
    users = []

    def fake_get_user_by_name(db, name):
        return next((u for u in users if u["name"] == name), None)

    def fake_create_user(db, user):
        created = {"id": len(users) + 1, "name": user.name, "email": user.email}
        users.append(created)
        return created

    monkeypatch.setattr(crud, "get_user_by_name", fake_get_user_by_name)
    monkeypatch.setattr(crud, "create_user", fake_create_user)

    response = client.post(
        "/users",
        json={"name": "prabhati", "email": "prabhati@example.com"},
    )

    assert response.status_code == 201
    assert response.json()["name"] == "prabhati"
    assert response.json()["email"] == "prabhati@example.com"


def test_create_user_duplicate(monkeypatch):
    users = [{"id": 1, "name": "prabhati", "email": "prabhati@example.com"}]

    def fake_get_user_by_name(db, name):
        return next((u for u in users if u["name"] == name), None)

    monkeypatch.setattr(crud, "get_user_by_name", fake_get_user_by_name)

    response = client.post(
        "/users",
        json={"name": "prabhati", "email": "prabhati@example.com"},
    )

    assert response.status_code == 409


def test_read_user_found(monkeypatch):
    def fake_get_user(db, user_id):
        return {"id": 1, "name": "prabhati", "email": "prabhati@example.com"}

    monkeypatch.setattr(crud, "get_user", fake_get_user)

    response = client.get("/users/1")
    assert response.status_code == 200
    assert response.json()["name"] == "prabhati"


def test_read_user_not_found(monkeypatch):
    def fake_get_user(db, user_id):
        return None

    monkeypatch.setattr(crud, "get_user", fake_get_user)

    response = client.get("/users/1")
    assert response.status_code == 404


def test_chat_anonymous(monkeypatch):
    def fake_generate_response(prompt):
        return "Hello from test"

    def fake_create_chat_message(db, user_id, prompt, response):
        return None

    monkeypatch.setattr(ai, "generate_response", fake_generate_response)
    monkeypatch.setattr(crud, "create_chat_message", fake_create_chat_message)

    response = client.post("/chat", json={"prompt": "Hi"})
    assert response.status_code == 200
    assert response.json()["reply"] == "Hello from test"


def test_chat_user_not_found(monkeypatch):
    def fake_get_user(db, user_id):
        return None

    monkeypatch.setattr(crud, "get_user", fake_get_user)

    response = client.post("/chat", json={"prompt": "Hi", "user_id": 99})
    assert response.status_code == 404
