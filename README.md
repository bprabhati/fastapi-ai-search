# fastapi-ai-search

A production-style FastAPI application with PostgreSQL, Docker, and OpenAI integration.

## Local setup

1. Install dependencies:

   ```powershell
   cd fastapi-rag-app
   python -m pip install poetry
   poetry install
   ```

2. Create a `.env` file in the project root and set your environment values:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql://postgres:postgres@db:5432/fastapi_chat
   ```

3. Start local services:

   ```powershell
   docker compose up --build
   ```

4. Open the API docs:

   ```text
   http://localhost:8000/docs
   ```

## Endpoints

- `GET /health`
- `POST /users`
- `GET /users/{user_id}`
- `POST /chat`

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

## AWS deployment overview

Recommended architecture:

- AWS ECR for Docker image registry
- AWS ECS Fargate for container deployment
- AWS RDS PostgreSQL for the database
- AWS CloudWatch Logs for monitoring
- AWS Secrets Manager or ECS environment variables for secrets

## ECS configuration

The sample task definition is in `ecs/task-definition.json`.
Update the following values before use:

- `REPLACE_WITH_ECR_IMAGE`
- `REPLACE_WITH_AWS_REGION`

Then use the task definition in your ECS service.
