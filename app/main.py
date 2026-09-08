"""FastAPI app: upload PDFs, inspect facts, evidence, and relations.

    uvicorn app.main:app --reload

The assignment's hard requirement is that new PDFs can be uploaded and the
results inspected, so upload is a real endpoint backed by the same pipeline the
CLI uses. Ingest runs in a background task because a 100-page filing takes
minutes, and a browser should not be holding a request open for that.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.parsing.pdf_parser import document_id
from app.pipeline import IngestError, ingest_pdf, rebuild_relations
from app.store import get_store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Veritas — Fact Knowledge Layer",
    description="Extracts grounded facts from PDFs and reconciles them across documents.",
    version="0.1.0",
)

UPLOADS = Path("data/uploads")
UPLOADS.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).parent / "web"

# In-memory job registry. A prototype does not need a job queue, but the UI does
# need to know whether a long ingest is still running.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _run_ingest(job_id: str, path: Path, window_limit: int | None):
    _set_job(job_id, status="running", filename=path.name)
    try:
        summary = ingest_pdf(path, window_limit=window_limit)
        _set_job(job_id, status="done", result=summary)
    except IngestError as exc:
        _set_job(job_id, status="failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - surfaced to the client, not swallowed
        log.exception("ingest failed")
        _set_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


@app.post("/api/documents")
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile,
    window_limit: int | None = Query(
        None, description="Cap windows read per document. Lower is faster and cheaper."
    ),
):
    """Upload a PDF. Returns a job id immediately; poll /api/jobs/{id}."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF file.")

    target = UPLOADS / file.filename
    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    doc_id = document_id(target)
    store = get_store()
    if store.has_document(doc_id):
        # Content-addressed, so a re-upload under any filename is recognised.
        return {
            "status": "already_ingested",
            "doc_id": doc_id,
            "message": "This document is already in the knowledge layer.",
        }

    job_id = uuid.uuid4().hex[:12]
    _set_job(job_id, status="queued", filename=file.filename, doc_id=doc_id)
    background.add_task(_run_ingest, job_id, target, window_limit)
    return {"status": "queued", "job_id": job_id, "doc_id": doc_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return job


@app.get("/api/documents")
def list_documents():
    return get_store().documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    store = get_store()
    if not store.has_document(doc_id):
        raise HTTPException(404, "No such document.")
    store.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


# --------------------------------------------------------------------------
# Facts and relations
# --------------------------------------------------------------------------


@app.get("/api/facts")
def list_facts(
    doc_id: str | None = None,
    search: str | None = None,
    min_confidence: float = 0.0,
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    facts = get_store().facts(
        doc_id=doc_id, search=search, min_confidence=min_confidence,
        limit=limit, offset=offset,
    )
    return [f.model_dump() for f in facts]


@app.get("/api/facts/{fact_id}")
def get_fact(fact_id: str):
    fact = get_store().fact(fact_id)
    if fact is None:
        raise HTTPException(404, "No such fact.")
    return fact.model_dump()


@app.get("/api/relations")
def list_relations(
    verdict: str | None = None,
    fact_id: str | None = None,
    key: str | None = None,
    search: str | None = None,
    cross_document: bool = False,
    min_confidence: float = 0.0,
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    """Relations, each returned with both facts inlined.

    The UI always needs the two sides and their evidence to render a judgement,
    so returning ids alone would force a request storm.
    """
    store = get_store()
    relations = store.relations(
        verdict=verdict, fact_id=fact_id, key=key, search=search,
        cross_document=cross_document,
        min_confidence=min_confidence, limit=limit, offset=offset,
    )
    out = []
    for rel in relations:
        left, right = store.fact(rel.left_id), store.fact(rel.right_id)
        if not (left and right):
            continue
        out.append({
            **rel.model_dump(),
            "left": left.model_dump(),
            "right": right.model_dump(),
        })
    return out


@app.post("/api/relations/rebuild")
def rebuild():
    """Re-adjudicate everything. Used after changing the reasoning rules."""
    return rebuild_relations()


@app.get("/api/stats")
def stats():
    store = get_store()
    docs = store.documents()
    return {
        "documents": len(docs),
        "facts": store.count_facts(),
        "verdicts": store.verdict_counts(),
        "pages": sum(d.get("page_count") or 0 for d in docs),
    }


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------


@app.get("/")
def index():
    page = STATIC / "index.html"
    if not page.exists():
        return JSONResponse({"error": "UI not built"}, status_code=404)
    return FileResponse(page)


@app.get("/app.js")
def script():
    return FileResponse(STATIC / "app.js", media_type="application/javascript")


@app.get("/app.css")
def styles():
    return FileResponse(STATIC / "app.css", media_type="text/css")
