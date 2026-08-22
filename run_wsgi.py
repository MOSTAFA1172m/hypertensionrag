"""
run_wsgi.py — production WSGI entrypoint.

Uses waitress (threaded, cross-platform, zero native deps) instead of Flask's
single-process dev server. Serves the app on 0.0.0.0:PORT (default 8000).

Run:
    python run_wsgi.py
"""

import os

from waitress import serve

import app as application

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    threads = int(os.getenv("WAITRESS_THREADS", "8"))
    print(f"* Serving on http://0.0.0.0:{port} (waitress, {threads} threads)")
    serve(application.app, host="0.0.0.0", port=port, threads=threads)