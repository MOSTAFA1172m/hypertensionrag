"""
app.py — Flask backend for the Hypertension RAG assistant.

Run:
    python app.py

Serves:
    GET  /                          -> index.html UI
    POST /api/query                 -> RAG answer with citation sources
    POST /api/upload                -> start async ingestion, returns upload_id
    GET  /api/upload-stream/<id>    -> SSE real-time pipeline progress
    GET  /api/guidelines            -> list indexed guidelines
    GET  /pdf/<filename>            -> serve raw PDF file
"""

import json
import queue
import sys
import threading
import uuid
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename

from ingestion.auto_ingest import auto_ingest_pdf
from rag.rag_pipeline import RAGResult, answer, get_bm25_retriever, get_qdrant_client

app = Flask(__name__)

PDF_DIR       = Path(__file__).parent / "data" / "raw" / "guidelines"
PDF_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = Path(__file__).parent / "data" / "source_registry.json"

# ── Active upload jobs: upload_id -> Queue of progress events ──
_upload_jobs: dict[str, queue.Queue] = {}


# ────────────────────────────────────────────────
# Static routes
# ────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/pdf/<path:filename>")
def serve_pdf(filename: str):
    return send_from_directory(PDF_DIR, filename, mimetype="application/pdf")


# ────────────────────────────────────────────────
# Guidelines registry
# ────────────────────────────────────────────────

@app.get("/api/guidelines")
def list_guidelines():
    """Return all currently indexed and available guideline documents."""
    guidelines = []
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                guidelines = json.load(f)
        except Exception:
            guidelines = []

    available_files = [f.name for f in PDF_DIR.glob("*.pdf")]
    return jsonify({"guidelines": guidelines, "pdf_files": available_files})


# ────────────────────────────────────────────────
# Upload — async ingestion with SSE progress
# ────────────────────────────────────────────────

@app.post("/api/upload")
def upload_pdf():
    """Save the uploaded PDF and start the ingestion pipeline in a background thread.
    Returns an upload_id immediately so the client can open an SSE stream."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        filename = f"guideline_{uuid.uuid4().hex[:8]}.pdf"

    save_path = PDF_DIR / filename
    file.save(str(save_path))

    org      = (request.form.get("organization")  or "Clinical Guideline").strip()
    doc_name = (request.form.get("document_name") or filename.replace("_", " ").replace(".pdf", "")).strip()

    # Create a progress queue and register job
    upload_id   = str(uuid.uuid4())
    prog_queue  = queue.Queue()
    _upload_jobs[upload_id] = prog_queue

    def _run():
        def _cb(stage: str, msg: str, pct: float):
            prog_queue.put({"stage": stage, "message": msg, "percent": pct})

        try:
            result = auto_ingest_pdf(
                pdf_path=save_path,
                document_name=doc_name,
                organization=org,
                qdrant_client=get_qdrant_client(),
                progress_callback=_cb,
            )
            get_bm25_retriever(force_reload=True)
            prog_queue.put({
                "stage":   "complete",
                "message": f"✓ Ingestion complete — {result['total_chunks']} chunks indexed",
                "percent": 100,
                "details": result,
            })
        except Exception as exc:
            prog_queue.put({
                "stage":   "error",
                "message": f"Ingestion failed: {str(exc)}",
                "percent": -1,
            })
        finally:
            # Sentinel to close the SSE stream
            prog_queue.put(None)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"upload_id": upload_id, "filename": filename})


@app.get("/api/upload-stream/<upload_id>")
def upload_stream(upload_id: str):
    """Server-Sent Events stream for real-time ingestion progress."""
    prog_queue = _upload_jobs.get(upload_id)
    if prog_queue is None:
        return jsonify({"error": "Unknown upload_id"}), 404

    @stream_with_context
    def _generate():
        try:
            while True:
                try:
                    event = prog_queue.get(timeout=60)
                except queue.Empty:
                    # Heartbeat to keep connection alive
                    yield "event: heartbeat\ndata: {}\n\n"
                    continue

                if event is None:
                    # Sentinel — done
                    yield "event: close\ndata: {}\n\n"
                    break

                yield f"data: {json.dumps(event)}\n\n"

                if event.get("stage") in ("complete", "error"):
                    break
        finally:
            _upload_jobs.pop(upload_id, None)

    return Response(
        _generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ────────────────────────────────────────────────
# RAG Query
# ────────────────────────────────────────────────

@app.post("/api/query")
def query():
    data    = request.get_json(silent=True) or {}
    q       = (data.get("query") or "").strip()
    history = data.get("history") or []

    if not q:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        result: RAGResult = answer(q, history=history if history else None)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "answer": result.answer,
        "sources": [
            {
                "chunk_id":      s.chunk_id,
                "document_id":   s.document_id,
                "pdf_file":      s.pdf_file,
                "section_title": s.section_title,
                "page_start":    s.page_start,
                "page_end":      s.page_end,
                "score":         s.score,
                "text":          s.text,
            }
            for s in result.sources
        ],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
