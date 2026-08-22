# Deploy on PythonAnywhere (free, no card) — separate experiment

Runs the **full Flask app with the original `index.html` UI** (sidebar,
conversation history, PDF viewer, clickable citations) on PythonAnywhere's free
tier: always-on, no sleep, no ZeroGPU quota, persistent uploads.

The Gradio Space stays as the primary demo; this is an independent deployment.

---

## Phase 1 — Account

1. Go to **https://www.pythonanywhere.com** → **Sign up** (free, no card).
2. Verify your email and log in. Your site will be
   `https://<username>.pythonanywhere.com`.

## Phase 2 — Create the web app

1. Click the **Web** tab → **Add a new web app**.
2. **Next** → choose **Manual configuration** → pick **Python 3.12** → **Next**.
3. Make a note of the **WSGI configuration file** path it shows
   (e.g. `/var/www/<username>_pythonanywhere_com_wsgi.py`).

## Phase 3 — Upload the project

1. On your PC, the deploy bundle is at:
   `C:\Users\mosta\Downloads\RAG\RAG\deploy.tar.gz`
   (rebuild it with the `tar` command in `DEPLOY.md` after code changes).
2. On PythonAnywhere: **Files** tab → navigate to `/home/<username>` →
   **Upload a file** → select `deploy.tar.gz`.
3. Open a console: **Consoles** tab → **Bash** → run:
   ```bash
   cd /home/<username>
   mkdir -p mysite
   tar -xzf deploy.tar.gz -C mysite
   cd mysite
   rm -f Dockerfile docker-compose.yml entrypoint.sh run_wsgi.py gradio_app.py SPACES_README.md
   ls        # should show app.py, src/, templates/, data/, requirements-flask.txt ...
   ```

## Phase 4 — Virtualenv + dependencies

In the same Bash console:

```bash
cd /home/<username>/mysite
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-flask.txt
```

## Phase 5 — WSGI file

1. **Web** tab → under **Code**, click the **WSGI configuration file** link.
2. Replace its contents with the content of
   `C:\Users\mosta\Downloads\RAG\RAG\pythonanywhere_wsgi.py`
   (and change `<username>` to yours).
3. **Save**.

## Phase 6 — API key (environment variable)

1. **Web** tab → scroll to **Environment variables** (below Code) → **Add**:
   - name: `GEMINI_API_KEY`
   - value: your key (from `C:\Users\mosta\Downloads\RAG\RAG\.env`, starts with `AQ.Ab8...`)
2. **Save**.

## Phase 7 — Reload & verify

1. **Web** tab → click the green **Reload** button (top).
2. Open **https://<username>.pythonanywhere.com** — the full Flask UI loads.
3. Ask a question; answers appear with clickable citations and the PDF viewer.
4. Errors → **Web** tab → **Error log**.

---

## Notes

- Free tier: 512 MB RAM, always on, daily CPU limit, 1 web app. Light chat use
  fits easily.
- Data (uploads, Qdrant, regenerated artifacts) persists under
  `~/mysite/data`.
- The admin **Upload Guideline PDF** works; SSE progress may only update at the
  end on PythonAnywhere.
- Updating: re-upload `deploy.tar.gz`, `tar -xzf` into `mysite`,
  `pip install -r requirements-flask.txt` if deps changed, then **Reload**.