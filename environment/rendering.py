"""
3D visualisation for AgriGuardEnv.

Primary renderer: a Three.js scene (assets/viewer.html) served by a tiny
stdlib HTTP server. The browser polls GET /state for the live field snapshot,
so the same JSON contract serves both the visualiser and any downstream
mobile/dashboard client.

Design note: the renderer never imports torch and never touches the env's
internals beyond `snapshot()`, so training runs at full speed with rendering off.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"


class _StateHandler(SimpleHTTPRequestHandler):
    """Serves assets/ statically and the live snapshot at /state."""

    def __init__(self, *args, store=None, **kwargs):
        self._store = store
        super().__init__(*args, directory=str(ASSETS), **kwargs)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        if self.path.startswith("/state"):
            body = json.dumps(self._store.get("frame") or {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", ""):
            self.path = "/viewer.html"
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        """Receive a PNG captured from the WebGL canvas and save it.

        This is how the environment figure in the report is produced, so the
        visual can be regenerated exactly rather than screenshotted by hand.
        """
        if not self.path.startswith("/capture"):
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(n).decode()
        out = Path(self._store.get("capture_path") or (ASSETS / "environment_view.png"))
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            body = json.loads(payload)
            data, points = body["png"], body.get("points")
        except (json.JSONDecodeError, KeyError):
            data, points = payload, None        # plain data-URL fallback
        out.write_bytes(base64.b64decode(data.split(",", 1)[-1]))
        if points:
            # Screen-space positions of each plot, projected by the renderer,
            # so the annotated report figure never relies on guessed pixels.
            out.with_suffix(".points.json").write_text(json.dumps(points, indent=2))
        self._store["captured"] = str(out)
        print(f"[renderer] environment figure saved -> {out}")
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def log_message(self, *args):
        pass  # keep the training terminal clean


def _free_port(preferred: int = 8420) -> int:
    for port in range(preferred, preferred + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 0


class ThreeJSRenderer:
    """Live 3D renderer. Call `update(env)` once per step."""

    def __init__(self, open_browser: bool = True, port: int | None = None,
                 title: str = "AgriGuard", capture_path: str | Path | None = None):
        self._store: dict = {"frame": None, "capture_path": capture_path,
                             "captured": None}
        handler = partial(_StateHandler, store=self._store)
        # Fall back to the next free port if the requested one is taken, so a
        # leftover viewer from an earlier run never blocks a new one.
        try:
            self.port = port or _free_port()
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        except OSError:
            self.port = _free_port(self.port + 1)
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self.port}/viewer.html"
        print(f"[renderer] {title} 3D viewer live at {self.url}")
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                print("[renderer] open the URL above in a browser")

    def update(self, env, extra: dict | None = None):
        frame = env.snapshot()
        if extra:
            frame.update(extra)
        self._store["frame"] = frame

    @property
    def captured(self) -> str | None:
        """Path of the PNG the viewer uploaded, once a capture has happened."""
        return self._store.get("captured")

    def close(self):
        self._server.shutdown()
        self._server.server_close()

    # context-manager sugar so play.py can use `with`
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def record_episode(frames: list[dict], path: str | Path) -> Path:
    """Persist an episode's frames so the viewer can replay it offline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frames))
    print(f"[renderer] recorded {len(frames)} frames -> {path}")
    return path
