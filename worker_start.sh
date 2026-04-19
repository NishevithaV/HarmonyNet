#!/bin/sh
# Bind a port so Render doesn't kill this Web Service.
# serve_forever() keeps the Python process alive in the background.
python -c "
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')
    def log_message(self, *a):
        pass

port = int(os.environ.get('PORT', 8080))
print(f'[worker_start] Health check on port {port}', flush=True)
HTTPServer(('', port), H).serve_forever()
" &

exec celery -A api.celery_app worker --loglevel=info --concurrency=1
