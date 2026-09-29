# EagleEye backend

This FastAPI service keeps the AI platform key on the server and returns a typed,
EagleEye-compatible decision-tree draft. The browser never receives the key.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `AI_PROVIDER` plus the corresponding
   provider credentials. The backend loads this file automatically; environment
   variables supplied by the deployment environment still take precedence.
4. Start the service: `uvicorn main:app --reload --port 8000`

The API is available at `http://localhost:8000`; its health endpoint is
`GET /health`. EagleEye calls `POST /decision-trees/generate` when it is served
from `http://localhost:3000`. The default CORS configuration also allows the
common local development ports 5050 and 5500, including their `127.0.0.1`
variants. For another origin, set `CORS_ORIGINS` to a comma-separated list.

Set `AI_PROVIDER=ai_platform` to use the existing platform configuration, or set
`AI_PROVIDER=soclaas` with `SOCLAAS_API_KEY`, `SOCLAAS_BASE_URL`, and
`SOCLAAS_MODEL`. The service keeps the selected key on the server and calls the
provider's OpenAI-compatible `/chat/completions` endpoint. It sends `x-api-key`
for `ai_platform` and `Authorization: Bearer ...` for SOCLAAS, then requests
structured JSON output for the decision-tree generator.
