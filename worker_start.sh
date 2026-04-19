#!/bin/sh
# Render Web Service requires a bound port. Start a minimal health-check server
# on $PORT alongside the Celery worker so Render doesn't kill the process.
python -c "
import os, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')
    def log_message(self, *a):
        pass

port = int(os.environ.get('PORT', 8080))
server = HTTPServer(('', port), H)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f'[worker_start] Health check listening on port {port}', flush=True)
" &

exec celery -A api.celery_app worker --loglevel=info --concurrency=1
