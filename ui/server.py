"""Thin FastAPI wrapper for the Hiver suggested-response engine."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hiver_email.data import load_pairs, sample_heldout, split_pairs
from hiver_email.generator import SuggestedReplyGenerator
from hiver_email.judge import RubricJudge
from hiver_email.llm import LLMClient
from hiver_email.retrieval import baseline_reply, build_retriever


UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
STATIC_DIR = UI_DIR / "static"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
RETRIEVER_KIND = "tfidf"


class SuggestRequest(BaseModel):
    email: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class JudgeRequest(BaseModel):
    email: str = Field(min_length=1)
    human_reply: str
    candidate_reply: str


def _retrieved_payload(examples: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "example_id": example.example_id,
            "incoming_email": example.incoming_email,
            "human_reply": example.human_reply,
            "similarity": example.similarity,
        }
        for example in examples
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    pairs = load_pairs()
    corpus, heldout = split_pairs(pairs)
    retriever = build_retriever(corpus, kind=RETRIEVER_KIND)
    client = LLMClient()

    app.state.corpus = corpus
    app.state.heldout = heldout
    app.state.retriever = retriever
    app.state.client = client
    app.state.generator = SuggestedReplyGenerator(retriever, client, top_k=3)
    app.state.judge = RubricJudge(client)
    yield


app = FastAPI(title="Hiver Email Suggested-Response UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "corpus_size": len(app.state.corpus),
        "gen_model": app.state.client.gen_model,
        "retriever": RETRIEVER_KIND,
    }


@app.post("/api/suggest")
def suggest(request: SuggestRequest) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        generator = SuggestedReplyGenerator(
            app.state.retriever, app.state.client, top_k=request.top_k
        )
        result = generator.generate(request.email)
    except Exception as exc:  # keep network/runtime failures visible to the UI
        return {
            "suggested_reply": "",
            "baseline_reply": "",
            "retrieved": [],
            "model": app.state.client.gen_model,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "error": str(exc),
        }

    elapsed = round(time.perf_counter() - started, 2)
    if result.error:
        return {
            "suggested_reply": "",
            "baseline_reply": "",
            "retrieved": [],
            "model": result.model,
            "elapsed_seconds": elapsed,
            "error": result.error,
        }
    return {
        "suggested_reply": result.reply,
        "baseline_reply": baseline_reply(result.retrieved),
        "retrieved": _retrieved_payload(result.retrieved),
        "model": result.model,
        "elapsed_seconds": elapsed,
        "error": "",
    }


@app.post("/api/judge")
def judge(request: JudgeRequest) -> dict[str, Any]:
    try:
        retrieved = app.state.retriever.retrieve(request.email, top_k=3)
        verdict = app.state.judge.judge(
            request.email,
            request.human_reply,
            request.candidate_reply,
            retrieved,
        )
        return {
            "task_fulfillment": verdict.task_fulfillment,
            "action_alignment": verdict.action_alignment,
            "completeness": verdict.completeness,
            "tone": verdict.tone,
            "critical_error": verdict.critical_error,
            "acceptable": verdict.acceptable,
            "judge_score": verdict.score_100(),
            "reason": verdict.reason,
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/examples")
def examples(n: int = Query(default=8, ge=1, le=50)) -> list[dict[str, str]]:
    return [
        {
            "example_id": pair.example_id,
            "incoming_email": pair.incoming_email,
            "human_reply": pair.human_reply,
        }
        for pair in sample_heldout(app.state.heldout, n)
    ]


@app.get("/api/metrics")
def metrics() -> Response:
    if not METRICS_PATH.exists():
        return JSONResponse({"available": False})
    try:
        json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse({"available": False, "error": str(exc)})
    return Response(METRICS_PATH.read_bytes(), media_type="application/json")
