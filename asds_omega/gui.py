"""Native Desktop GUI Launcher for the ASDS-OMEGA v2.0 Scientific Workbench.

Architecture:
    Starts the local FastAPI scientific compute engine on an ephemeral loopback
    port (`127.0.0.1`) and opens the workbench as a standalone desktop window:
      1. Uses `pywebview` (native OS WebView2 / Cocoa / GTK) if installed.
      2. Otherwise launches Edge or Chrome in frameless `--app=` desktop mode
         (zero browser tabs or address bar, dedicated OS window & taskbar icon).
      3. Falls back cleanly to the default system browser if neither is found.

This gives scientists a native desktop experience on Windows/macOS/Linux while
keeping the exact same KaTeX math engine, WebGL 3D plots, and zero-bloat
single-codebase architecture used in headless server/Docker deployments.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_server(url: str, timeout: float = 12.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/datasets", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def _find_chromium_app_binary() -> str | None:
    """Locate Edge or Chrome for frameless standalone `--app=` desktop window mode."""
    candidates: list[str] = []
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    else:
        for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
            found = shutil.which(name)
            if found:
                candidates.append(found)

    for path in candidates:
        if Path(path).exists():
            return path
    return None


def launch_desktop_gui(host: str = "127.0.0.1", port: int | None = None) -> None:
    """Start the ASDS-OMEGA v2.0 scientific backend and open the desktop workbench window."""
    import uvicorn

    chosen_port = port if port is not None else _find_free_port(host)
    url = f"http://{host}:{chosen_port}"

    config = uvicorn.Config("app.server:app", host=host, port=chosen_port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if not _wait_for_server(url):
        raise RuntimeError(f"ASDS-OMEGA scientific server failed to start on {url}")

    # Priority 1: Native pywebview desktop window if installed
    try:
        import webview  # type: ignore[import-not-found]

        webview.create_window(
            "ASDS-OMEGA v2.0 — Scientific Discovery Workbench",
            url,
            width=1480,
            height=920,
            min_size=(1100, 720),
        )
        webview.start()
        server.should_exit = True
        return
    except ImportError:
        pass

    # Priority 2: Standalone frameless desktop window via Edge / Chrome --app
    app_bin = _find_chromium_app_binary()
    if app_bin is not None:
        print(f"[ASDS-OMEGA GUI] Opening standalone desktop workbench at {url}", flush=True)
        proc = subprocess.Popen(
            [app_bin, f"--app={url}", "--window-size=1480,920", "--new-window"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        finally:
            server.should_exit = True
        return

    # Priority 3: Default system browser
    print(f"[ASDS-OMEGA GUI] Serving workbench at {url} (Ctrl+C to stop)...", flush=True)
    webbrowser.open(url)
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
