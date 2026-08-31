FastAPI → WordPress setup

1. Install

pip install "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1"

2. Local environment

Copy .env.example to .env and configure it.

The project already uses .env for the LLM configuration. Keep the
chatbot's existing environment variables there too.

3. Run

python -m uvicorn api:app --reload

4. Test

Health:

http://127.0.0.1:8000/health

Swagger:

http://127.0.0.1:8000/docs

5. WordPress CORS

When the WordPress site is ready, change:

CORS_ORIGINS=https://www.zyraluxe.com,https://zyraluxe.com

Do not leave allow_origins=["*"] in a production API.

6. Streaming contract

POST /chat/stream returns:

event: session
data: <session-id>

data: first chunk

data: second chunk

event: done
data: [DONE]

If an unexpected server/LLM error occurs:

event: error
data: {"message":"Unable to complete the response."}

The WordPress JavaScript widget will consume this SSE stream.

7. Session behavior

First request:

{
  "message": "show me gold rings under 30000"
}

The API returns a session event.

Next request:

{
  "message": "which one is cheapest?",
  "session_id": "<previous-session-id>"
}

This preserves the current chatbot conversation.

8. Important production security notes

CORS

CORS controls which browser origins can call the API. It is not authentication.

API keys

Do not place a secret API key in public browser JavaScript. Anyone can inspect it.

If the WordPress widget calls FastAPI directly, protect the deployment with:

HTTPS

reverse proxy/WAF

rate limiting

origin restrictions

server-side abuse controls

If WordPress proxies requests to FastAPI, the secret key can stay server-side.

Sessions

The current dictionary is intentionally for development.

For a real deployment with multiple workers or containers, move
conversation state to Redis or another shared store.

Streaming + validation

The current chatbot streams the generated answer and validates the
complete answer after generation. That means a streamed answer can reach
the client before final validation.

For strict production safety, we should change this before launch:
either validate before releasing product/knowledge output, or introduce
a structured/claim-safe streaming design.

The API layer does not silently claim post-stream validation is a
security boundary.