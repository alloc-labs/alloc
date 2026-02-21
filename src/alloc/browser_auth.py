"""OAuth PKCE browser login for Alloc CLI.

Opens the user's browser for OAuth (Google/Microsoft) and captures tokens
via a local HTTP callback server. Industry-standard pattern used by gh,
gcloud, stripe, and vercel CLIs.

No external dependencies — uses only Python stdlib.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import socket
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs


def _generate_pkce_pair():
    """Generate a PKCE code_verifier and code_challenge (S256).

    Returns (verifier, challenge) tuple.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _find_open_port(start=17256, attempts=20):
    """Find an available TCP port starting from *start*.

    Tries *attempts* consecutive ports and returns the first one available.
    Raises RuntimeError if none are free.
    """
    for port in range(start, start + attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No open port found in range {start}-{start + attempts - 1}")


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback code."""

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/callback" and "code" in params:
            self.server.auth_code = params["code"][0]
            self._respond(
                200,
                "<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
                "<h2>Login successful</h2>"
                "<p>You can close this tab and return to your terminal.</p>"
                "</body></html>",
            )
        elif parsed.path == "/callback" and "error" in params:
            error_desc = params.get("error_description", params.get("error", ["unknown error"]))[0]
            self.server.auth_error = error_desc
            self._respond(
                400,
                "<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
                f"<h2>Login failed</h2><p>{error_desc}</p>"
                "</body></html>",
            )
        else:
            self._respond(404, "Not found")

    def _respond(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress request logging to keep terminal clean.
        pass


def browser_login(
    provider,
    supabase_url,
    anon_key,
    timeout_seconds=120,
):
    """Run the full OAuth PKCE browser login flow.

    1. Generate PKCE pair
    2. Start local HTTP server on 127.0.0.1
    3. Open browser to Supabase authorize URL
    4. Wait for callback with auth code
    5. Exchange code for tokens via Supabase token endpoint
    6. Return dict with access_token, refresh_token, email

    Raises RuntimeError on timeout or auth failure.
    """
    import httpx

    verifier, challenge = _generate_pkce_pair()
    port = _find_open_port()

    redirect_uri = f"http://localhost:{port}/callback"

    authorize_params = urlencode({
        "provider": provider,
        "redirect_to": redirect_uri,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    authorize_url = f"{supabase_url}/auth/v1/authorize?{authorize_params}"

    server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.auth_code = None  # type: ignore[attr-defined]
    server.auth_error = None  # type: ignore[attr-defined]
    server.timeout = 1  # poll interval for handle_request()

    # Run server in a thread so we can enforce the overall timeout.
    server_thread = threading.Thread(target=_serve_until_done, args=(server, timeout_seconds))
    server_thread.daemon = True
    server_thread.start()

    # Open the browser (or print URL as fallback).
    try:
        opened = webbrowser.open(authorize_url)
    except Exception:
        opened = False

    if not opened:
        print(f"\nOpen this URL in your browser to log in:\n\n  {authorize_url}\n")
    else:
        print("Opened browser for login. Waiting for callback...")

    server_thread.join(timeout=timeout_seconds + 5)
    server.server_close()

    if server.auth_error:
        raise RuntimeError(f"OAuth error: {server.auth_error}")

    if not server.auth_code:
        raise RuntimeError("Login timed out — no callback received within 120 seconds.")

    # Exchange auth code + verifier for tokens.
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            f"{supabase_url}/auth/v1/token?grant_type=pkce",
            json={
                "auth_code": server.auth_code,
                "code_verifier": verifier,
            },
            headers={
                "apikey": anon_key,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    access_token = data.get("access_token", "")
    refresh_token = data.get("refresh_token", "")
    email = data.get("user", {}).get("email", "")

    if not access_token:
        raise RuntimeError("Token exchange failed: no access token received.")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "email": email,
    }


def _serve_until_done(server, timeout_seconds):
    """Handle requests until auth_code/auth_error is set or timeout expires."""
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.auth_code or server.auth_error:
            break
        server.handle_request()
