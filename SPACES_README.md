---
title: Hypertension Guidelines RAG
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Hypertension Guidelines RAG

Retrieval-augmented generation over WHO, ESC, USPSTF, and CDC hypertension
guidelines. Users ask questions and get citation-backed answers (Gemini + BM25);
admins can upload additional guideline PDFs to index.

- `app.py` in this repo is the Gradio app (`gradio_app.py` in the source project).
- Set `GEMINI_API_KEY` as a Space secret (Settings -> Variables and secrets).
- Free tier sleeps after ~48 h of inactivity and wakes on the next request.