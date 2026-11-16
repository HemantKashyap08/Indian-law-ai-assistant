import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from app.rag import analyze

BASE = Path(__file__).resolve().parent.parent
log = logging.getLogger("nyayamitra")


def client_ip(request: Request) -> str:
    # Behind a host's proxy the real visitor IP is in X-Forwarded-For
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=client_ip)
app = FastAPI(title="NyayaMitra", docs_url=None, redoc_url=None)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def too_many(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a minute and try again."},
    )


class Turn(BaseModel):
    role: str
    text: str = Field(max_length=3000)


class AnalyzeRequest(BaseModel):
    case: str = Field(min_length=3, max_length=4000)
    incident_date: str | None = None
    history: list[Turn] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
@limiter.limit("300/day", key_func=lambda request: "all-visitors")  # protects your Gemini quota
@limiter.limit("6/minute;60/day")  # per visitor
def api_analyze(request: Request, req: AnalyzeRequest):
    try:
        return analyze(
            req.case,
            req.incident_date,
            [t.model_dump() for t in req.history][-6:],
        )
    except Exception:
        log.exception("analyze failed")
        raise HTTPException(502, "Could not analyse right now. Please try again.")


app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")