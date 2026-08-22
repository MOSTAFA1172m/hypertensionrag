# Deploy on Hugging Face Spaces (free, no credit card)

The free path. No card required. HF now charges for **Docker** Spaces, so we use
a **Gradio** Space (free CPU-2x / 16 GB RAM): the HF runtime installs
`requirements.txt` and runs `app.py` (our Gradio UI — same RAG pipeline as the
Flask app). It sleeps after ~48 h idle and wakes on the next request (~30 s cold
start). The BM25 corpus is shipped with the repo, so Q&A always works even
though uploads/Qdrant reset on restart.

---

## Phase 1 — Create the Space

1. Create a free account at <https://huggingface.co/join> (no card).
2. Go to <https://huggingface.co/new-space>:
   - Space name: `hypertension-rag`
   - License: `mit`
   - **SDK: Gradio**
   - Click **Create Space**.
3. Create a **User Access Token** at <https://huggingface.co/settings/tokens>
   (type: Write). This is your git password for pushes.

## Phase 2 — Push the app

From your local machine:

```bash
git clone https://huggingface.co/spaces/<your-username>/hypertension-rag
cd hypertension-rag
```

Copy the app into the Space repo. From the project root, using the deploy tarball:

```powershell
# PowerShell, project root:
tar -xzf deploy.tar.gz -C ..\hypertension-rag
# the Space runs app.py — swap the Flask app for the Gradio app:
Copy-Item ..\RAG\gradio_app.py ..\hypertension-rag\app.py
# and use the Space README (frontmatter sdk: gradio):
Copy-Item ..\RAG\SPACES_README.md ..\hypertension-rag\README.md
```

Then push:

```bash
git add -A
git commit -m "deploy rag"
git push            # username = your HF username, password = the Access Token
```

## Phase 3 — Set the secret

Space **Settings → Variables and secrets → New secret**:
- Key: `GEMINI_API_KEY`
- Value: your key

## Phase 4 — Verify

- The runtime installs deps and starts automatically. Watch the Space **Builder**
  tab; then open the app at `https://<your-username>-hypertension-rag.hf.space`.
- Ask a question in the chat (e.g. "What is the BP threshold for pharmacological
  treatment?") — you should get a cited answer.
- The **Admin — Upload Guideline** tab ingests a new PDF (takes a few minutes).

---

# Appendix — Oracle Cloud Always Free (optional)

Same image but a 24/7 VM. Needs a card for identity verification only (never
charged). Skip if you don't want to provide card details. This path uses the
Flask app + Docker (Dockerfile/compose/run_wsgi.py) — not the Gradio app.

1. Sign up at <https://signup.cloud.oracle.com> (choose "Always Free").
2. **Compute → Instances → Create instance**: Ubuntu 24.04, Ampere A1 shape
   (4 OCPU / 24 GB free, or the free `VM.Standard.E2.1.Micro` x86).
3. Add SSH key (download the generated private key). Note the **Public IP**.
4. Open port **8000**: VCN → Security List → Add Ingress Rule (TCP, 8000, 0.0.0.0/0).
5. SSH in and install Docker:
   ```bash
   ssh -i <key>.pem ubuntu@<IP>
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER && newgrp docker
   ```
6. Upload and deploy:
   ```powershell
   # local
   scp deploy.tar.gz ubuntu@<IP>:/home/ubuntu/
   ```
   ```bash
   # VM
   cd ~ && tar -xzf deploy.tar.gz
   cat > .env <<'EOF'
   GEMINI_API_KEY=<your-key>
   EOF
   docker compose up -d --build
   ```
7. App at `http://<IP>:8000`. `docker-compose.yml` sets `PORT=8000` and a
   persistent `./data` volume for uploads.

---

## Updating the app

- HF Spaces: push new commits (or re-copy `gradio_app.py` → `app.py`); the Space
  rebuilds automatically.
- Oracle: `scp deploy.tar.gz` again, then on the VM:
  ```bash
  tar -xzf deploy.tar.gz -C ~
  docker compose build rag && docker compose up -d
  ```

## Notes

- `deploy.tar.gz` rebuild command (PowerShell, project root) — excludes secrets,
  venv, and heavy data:
  ```powershell
  tar -czf deploy.tar.gz --exclude=myenv --exclude=.env --exclude=evaluation `
    --exclude=notebooks --exclude=assets --exclude=__pycache__ --exclude="*.pyc" `
    --exclude=deploy.tar.gz --exclude=data/qdrant --exclude=data/raw `
    --exclude=data/qdrant_eval --exclude=data/qdrant_ablation `
    --exclude=data/processed/embeddings.json --exclude=data/processed/embeddings_dedup.json `
    --exclude=data/processed/embeddings_pubmedbert_dedup.npy `
    --exclude=data/processed/medcpt_article_vectors.npy `
    --exclude=data/processed/validated_chunks.json `
    --exclude=data/processed/chunk_remap_dedup.json `
    --exclude=data/processed/chunks.json --exclude=data/processed/cleaned_pages.json `
    --exclude=data/processed/extracted_pages.json --exclude=data/processed/sections.json `
    --exclude=data/processed/validation_report.json .
  ```
- Gradio Space = Gradio runtime (`app.py` + `requirements.txt`); the Docker files
  are only used by the Oracle path.
- Local smoke test: `python gradio_app.py` → http://127.0.0.1:7860 (Gradio must
  be installed: `pip install gradio`).
- If the chat returns an error, the `GEMINI_API_KEY` secret is missing/wrong —
  check the Space logs.