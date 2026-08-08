# FastAPI RAG App

A production-style FastAPI application with PostgreSQL, Docker, and a basic GenAI chatbot endpoint.

## Local setup

1. Install dependencies:

   ```powershell
   cd fastapi-rag-app
   python -m pip install poetry
   poetry install
   ```

2. Create a `.env` file and set your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Start services:

   ```powershell
   docker compose up --build
   ```

4. Visit `http://localhost:8000/docs` for API docs.

## Endpoints

- `GET /health`
- `POST /users`
- `GET /users/{user_id}`
- `POST /chat`

## Notes

- The app uses PostgreSQL in Docker and a basic OpenAI chatbot integration.
- `DATABASE_URL` is configured in `docker-compose.yml`.
