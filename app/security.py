import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or len(JWT_SECRET) < 32:
    raise RuntimeError("Set JWT_SECRET to a random value of at least 32 characters.")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = 60
PASSWORD_HASH_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str | None) -> bool:
    if not encoded_hash:
        return False

    try:
        algorithm, iterations, salt_hex, expected_hex = encoded_hash.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(digest.hex(), expected_hex)
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
        "iss": "fastapi-rag-app",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    payload = jwt.decode(
        token,
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
        issuer="fastapi-rag-app",
        options={"require": ["sub", "exp", "iat", "iss"]},
    )
    subject = payload["sub"]
    if not subject.isdecimal():
        raise jwt.InvalidTokenError("Invalid token subject.")
    return int(subject)
