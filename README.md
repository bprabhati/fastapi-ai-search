# fastapi-ai-search

A FastAPI application with PostgreSQL, pgvector, Docker, and OpenAI integration.

## Local setup

1. Install dependencies:

   ```powershell
   cd fastapi-rag-app
   python -m pip install poetry
   poetry install
   ```

2. Create a `.env` file in the project root and set your environment values:

   ```env
   AI_PROVIDER=ollama
   OLLAMA_CHAT_MODEL=llama3.2:3b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   JWT_SECRET=paste_a_random_secret_here
   # Only required when AI_PROVIDER=openai
   OPENAI_API_KEY=
   ```

   Docker Compose supplies the container database URL automatically.
   `AI_PROVIDER=ollama` uses the local Ollama models without OpenAI API
   requests. Docker Compose connects to Ollama on
   the Windows host using `host.docker.internal:11434`. Set `AI_PROVIDER=openai`
   and provide a valid `OPENAI_API_KEY` to use OpenAI instead. Set
   `OPENAI_CHAT_MODEL` to choose the OpenAI model. Embeddings follow
   `AI_PROVIDER` unless you set `EMBEDDING_PROVIDER`, so `AI_PROVIDER=openai`
   with `EMBEDDING_PROVIDER=ollama` keeps your existing local embeddings. If you
   change the embedding model, set `EMBEDDING_DIMENSIONS` to match and call
   `POST /documents/reindex`. Generate a JWT
   signing secret in PowerShell with
   `python -c "import secrets; print(secrets.token_urlsafe(48))"` and paste the
   output as `JWT_SECRET`. Keep `.env` private and do not commit it.

3. Start local services:

   ```powershell
   docker compose up --build
   ```

4. Open the chat UI and create an account:

   ```text
   http://localhost:8000/
   ```

   Register with your name, email, and password, then sign in. The UI shows
   the signed-in account and lets you upload a PDF or DOCX and chat with your
   own indexed documents. Selecting a file starts the upload. Swagger remains
   available at `http://localhost:8000/docs`; use its **Authorize** control
   with a bearer token for protected endpoints.

   `AI_PROVIDER=mock` is only for testing the request flow and does not provide
   semantic answers. With Ollama, after signing in, call `POST /documents/reindex` to
   regenerate vectors for existing documents. This project uses the
   768-dimensional `nomic-embed-text` vectors; existing 1536-dimensional
   vectors are preserved but excluded from search until rebuilt.

## Endpoints

- `GET /health`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `GET /users/{user_id}` (only the signed-in user's own record)
- `POST /documents`
- `GET /documents`
- `POST /documents/reindex`
- `POST /documents/upload`
- `POST /chat`
- `POST /agent/chat`

Registration and login return an HS256 JWT bearer token that expires after
60 minutes. Document upload, listing, and chat endpoints require that token.
Document retrieval is scoped to the authenticated user. Existing documents
uploaded before login are unowned and are not automatically visible to new
accounts; upload them again after signing in.

`POST /documents/reindex` recomputes embeddings for the signed-in user's
existing document chunks using the configured embedding model. PostgreSQL stores variable-dimension vectors so
OpenAI's 1536-dimensional embeddings and Ollama's 768-dimensional embeddings
can both be stored; retrieval compares only vectors of the active model's
dimension.

### JWT learning flow

1. Register with `POST /auth/register` or sign in with `POST /auth/login`.
2. The response includes `access_token` and the user profile. The UI stores
   the token in the current browser tab and shows the signed-in user's name.
3. For protected API calls, send `Authorization: Bearer <access_token>`.
   FastAPI verifies the HMAC signature, issuer, and expiry, then loads the user
   from the token's `sub` claim.
4. Passwords are stored as salted PBKDF2-SHA256 hashes; the raw password is
   never stored. The JWT signing secret stays on the server in `JWT_SECRET`.

Logging out clears the browser's token. Tokens are otherwise stateless and
remain valid until their 60-minute expiry, so this starter does not implement
server-side token revocation or refresh tokens.

`POST /documents` creates a document, splits its content into chunks, and stores
their embeddings in PostgreSQL using pgvector. `POST /chat` retrieves similar
chunks before generating a response. It also uses up to six recent turns from
the signed-in account to resolve follow-up questions. Previous assistant
answers are conversational context only; retrieved document excerpts remain
the source of truth. This starter does not yet separate chat history into
individual conversations.

`POST /documents/upload` accepts a PDF or DOCX file as multipart form data,
extracts its text, and indexes it using the same chunking and embedding flow.
For PDF pages with no selectable text, the app uses local Tesseract OCR to read
the rendered page. It also OCRs embedded images on pages that contain
selectable text. Docker installs Tesseract automatically; rebuild the image
with `docker compose up --build` after adding OCR. When running the app
directly on Windows, install Tesseract OCR and ensure `tesseract.exe` is on
`PATH`.

`POST /agent/chat` uses OpenAI function calling to let the model decide whether
and how many times to call the `search_knowledge_base` / `list_documents` tools
before answering, instead of always doing one fixed retrieval step. Requires
`AI_PROVIDER=openai` and a real `OPENAI_API_KEY` to exercise actual tool calls;
with Ollama or mock mode it falls back to a single retrieval step like `/chat`.

## GitHub CI/CD

This repository includes a GitHub Actions workflow at `.github/workflows/deploy.yml`.
The workflow runs on push to `main` and performs:

- checkout
- install dependencies
- run tests
- build Docker image
- push image to Amazon ECR
- deploy to Amazon ECS

### GitHub secrets required

- `AWS_REGION`
- `AWS_ROLE_TO_ASSUME`
- `ECR_REGISTRY`
- `ECR_REPOSITORY`
- `ECS_CLUSTER`
- `ECS_SERVICE`

The workflow also needs access to the runtime `DATABASE_URL` and
`OPENAI_API_KEY`. Store these as ECS task-definition secrets backed by AWS
Secrets Manager or Systems Manager Parameter Store; do not commit them to this
repository.

## AWS deployment overview

Recommended architecture:

- AWS ECR for Docker image registry
- AWS ECS Fargate for container deployment
- AWS RDS PostgreSQL for the database
- AWS CloudWatch Logs for monitoring
- AWS Secrets Manager or ECS environment variables for secrets

## ECS configuration

The sample task definition is in `ecs/task-definition.json`.
Before deploying, configure the following values:

- `REPLACE_WITH_ECR_IMAGE`
- `REPLACE_WITH_AWS_REGION`
- the runtime database URL or ECS secret reference
- the OpenAI API key or ECS secret reference

The current workflow builds and pushes the Docker image, but the sample task
definition still contains the image placeholder. Replace it with the pushed
image URI, or add an ECS task-definition rendering step to the workflow before
the deploy step. ECS does not expand shell-style values such as
`${DATABASE_URL}` or `${OPENAI_API_KEY}` in the task definition.

For production, use Amazon RDS PostgreSQL with the pgvector extension enabled.
