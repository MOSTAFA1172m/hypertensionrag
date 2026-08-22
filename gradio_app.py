"""
gradio_app.py — free HF Spaces entry (Gradio SDK).

A Gradio port of the Flask app's UI (templates/index.html): same light
navy/teal theme, suggested questions, conversation chat with citation-backed
answers, and a source-card panel. Also keeps the Admin guideline upload tab.

In the Space repo this file is named app.py (the Gradio runtime runs app.py).
"""

import html
import os
import shutil
import sys
from pathlib import Path

import gradio as gr

# HF Spaces runtime requires handlers to be decorated with @spaces.GPU (the
# `spaces` package is preinstalled there). Falls back to a no-op locally.
try:
    import spaces
except ImportError:  # pragma: no cover - local dev without the HF runtime
    spaces = None


def _gpu(duration: int):
    def deco(fn):
        if spaces is not None:
            return spaces.GPU(duration=duration)(fn)
        return fn

    return deco


# src/ on the path (same convention as app.py) so ingestion.* and rag.* import
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion.auto_ingest import auto_ingest_pdf
from rag.rag_pipeline import answer, get_bm25_retriever, get_qdrant_client

DATA_DIR = Path(__file__).parent / "data"
PDF_DIR = DATA_DIR / "raw" / "guidelines"
MAX_TURNS = 6

SUGGESTIONS = [
    "What is the BP threshold for initiating pharmacotherapy?",
    "What are the target blood pressure goals for adults?",
    "What is the first-line medication for hypertension?",
    "When should antihypertensive therapy be started?",
]

NAVY = "#1a3a5c"
NAVY_LIGHT = "#2c5282"
TEAL = "#0891b2"
BG = "#f0f4f8"
BORDER = "#dde3ed"
MUTED = "#64748b"

CSS = f"""
.gradio-container {{ background: {BG} !important; max-width: 1200px !important; margin: 0 auto !important; }}
footer {{ display: none !important; }}
#titlebar {{ background: linear-gradient(90deg, #112540, {NAVY} 55%, {TEAL}); color: #fff; padding: 16px 22px; border-radius: 12px; margin: 6px 0 10px; }}
#titlebar h1 {{ margin: 0; font-size: 20px; color: #fff !important; }}
#titlebar p {{ margin: 3px 0 0; font-size: 12px; opacity: .85; color: #fff !important; }}
.source-card {{ border: 1px solid {BORDER}; border-left: 4px solid {TEAL}; border-radius: 8px; padding: 8px 12px; margin: 6px 0; background: #fff; }}
.source-card .src-doc {{ font-weight: 700; color: {NAVY}; font-size: 13px; }}
.source-card .src-sec {{ color: {NAVY_LIGHT}; font-size: 12px; }}
.source-card .src-quote {{ color: {MUTED}; font-size: 11px; font-style: italic; margin-top: 4px; }}
"""

TITLEBAR = f"""
<div id="titlebar">
  <h1>Hypertension Guidelines RAG</h1>
  <p>WHO &middot; ESC &middot; USPSTF &middot; CDC — grounded answers with citations. Not medical advice.</p>
</div>
"""


def _sources_html(sources) -> str:
    cards = []
    for i, s in enumerate(sources, 1):
        text = (s.text or "").strip()
        snippet = text[:180] + ("…" if len(text) > 180 else "")
        cards.append(
            '<div class="source-card">'
            f'<div class="src-doc">[{i}] {html.escape(s.document_name)}</div>'
            f'<div class="src-sec">{html.escape(s.section_title)} &middot; p. {s.page_start}</div>'
            f'<div class="src-quote">&ldquo;{html.escape(snippet)}&rdquo;</div>'
            "</div>"
        )
    return "\n".join(cards) or "<div style='color:#94a3b8'>No sources retrieved.</div>"


@_gpu(duration=45)
def _chat_fn(message: str, history: list):
    turns = []
    for turn in history[-MAX_TURNS:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            turns.append({"role": turn["role"], "content": turn.get("content", "")})

    try:
        result = answer(message, history=turns)
    except Exception as exc:  # e.g. GEMINI_API_KEY missing
        return (
            history
            + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"Error: {exc}"},
            ],
            "",
        )

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.answer},
    ]
    return history, _sources_html(result.sources)


def _file_path_and_name(file) -> tuple[Path | None, str | None]:
    if file is None:
        return None, None
    path = None
    orig = None
    if isinstance(file, dict):
        path, orig = file.get("path"), file.get("orig_name")
    else:
        path = getattr(file, "path", None) or getattr(file, "name", None)
        orig = getattr(file, "orig_name", None)
    if not path:
        return None, None
    return Path(path), (orig or Path(path).name)


@_gpu(duration=600)
def _ingest_pdf(pdf_path: str, orig_name: str, doc_name: str, org: str) -> str:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in orig_name if c.isalnum() or c in "._- ").strip() or "guideline.pdf"
    dest = PDF_DIR / safe
    shutil.copyfile(pdf_path, dest)

    try:
        result = auto_ingest_pdf(
            pdf_path=dest,
            document_name=(doc_name or None),
            organization=(org or "Clinical Guideline"),
            qdrant_client=get_qdrant_client(),
            progress_callback=None,
        )
        get_bm25_retriever(force_reload=True)
        return f"Ingestion complete — {result['total_chunks']} chunks indexed."
    except Exception as exc:
        return f"Ingestion failed: {exc}"


def _admin_upload(file, doc_name: str, org: str) -> str:
    path, orig = _file_path_and_name(file)
    if path is None:
        return "No file selected."
    if not orig.lower().endswith(".pdf"):
        return "Only PDF files are supported."
    # pass plain strings across the spaces.GPU boundary (FileData doesn't pickle)
    return _ingest_pdf(str(path), orig, doc_name or "", org or "")


with gr.Blocks(title="Hypertension Guidelines RAG") as demo:
    gr.HTML(TITLEBAR)

    with gr.Tab("Ask"):
        gr.Markdown(
            "Ask a question about the indexed hypertension guidelines. Answers are "
            "grounded in the guideline text and cite their sources."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460, show_label=False)
                msg = gr.Textbox(
                    placeholder="e.g. What is the BP threshold for pharmacological treatment?",
                    show_label=False,
                )
                gr.Examples(examples=SUGGESTIONS, inputs=msg, label="Suggested questions")
            with gr.Column(scale=2):
                gr.Markdown("#### Sources")
                sources_box = gr.HTML("")

        clear = gr.ClearButton([chatbot, sources_box], value="Clear chat")
        msg.submit(_chat_fn, [msg, chatbot], [chatbot, sources_box]).then(
            lambda: "", None, msg
        )

    with gr.Tab("Admin — Upload Guideline"):
        gr.Markdown(
            "Upload a guideline PDF to index it. Parsing + embedding a large PDF "
            "takes a few minutes."
        )
        file_in = gr.File(label="Guideline PDF", file_types=[".pdf"])
        doc_name = gr.Textbox(label="Document name (optional)")
        org = gr.Textbox(label="Organization (optional)", value="Clinical Guideline")
        ingest_btn = gr.Button("Ingest")
        out = gr.Markdown()

        ingest_btn.click(_admin_upload, [file_in, doc_name, org], out)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.cyan, neutral_hue=gr.themes.colors.slate
        ),
        css=CSS,
    )