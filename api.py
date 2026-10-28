"""
Minimal HTTP server: serves the dashboard and simulation state JSON files.
ponytail: stdlib http.server — no Flask, no extra dependency.
          Upgrade path: swap for Flask if POST routes or auth middleware are needed.

Usage: python api.py
       Open http://localhost:5000
"""
import http.server
import os

BASE = os.path.dirname(__file__)
DASHBOARD_DIR = os.path.join(BASE, "dashboard")
ROUTES = {
    "/state": os.path.join(BASE, "fl_state.json"),
    "/chain": os.path.join(BASE, "chain_state.json"),
}
PORT = 5000
MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript"}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence per-request logs

    def _ok(self, data: bytes, mime: str):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query params

        if path in ROUTES:
            try:
                with open(ROUTES[path], "rb") as f:
                    self._ok(f.read(), "application/json")
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
            return

        # Serve dashboard static files
        if path == "/" or path == "/index.html":
            filepath = os.path.join(DASHBOARD_DIR, "index.html")
        else:
            filepath = os.path.join(DASHBOARD_DIR, path.lstrip("/"))

        ext = os.path.splitext(filepath)[1]
        mime = MIME.get(ext, "text/plain")
        try:
            with open(filepath, "rb") as f:
                self._ok(f.read(), mime)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    server = http.server.HTTPServer(("", PORT), Handler)
    print(f"Dashboard -> http://localhost:{PORT}")
    print("    Run  python run_simulation.py  in a separate terminal to start the simulation.")
    print("    Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
