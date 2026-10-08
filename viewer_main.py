"""Deployed entry point for the database viewer (db_viewer.py).

Served from a public Cloud Run URL (scripts/deploy_db_viewer.sh), so every
request needs a session from a Google sign-in whose verified email is in
DB_VIEWER_ALLOWED_EMAILS. The OAuth consent screen is kept in Testing mode with
only those accounts as test users, so Google turns anyone else away first.

Configuration, all required (the app refuses to start without it):
  GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET  the web OAuth client
  DB_VIEWER_SECRET_KEY      signs the session cookie; changing it signs everyone out
  DB_VIEWER_ALLOWED_EMAILS  comma- or space-separated
"""

import logging
import os
from datetime import timedelta
from typing import Any, FrozenSet, Mapping, Optional

from authlib.integrations.base_client import (  # type: ignore[import-untyped]
    OAuthError,
)
from authlib.integrations.flask_client import OAuth  # type: ignore[import-untyped]
from flask import Flask, Response, abort, redirect, request, session, url_for
from flask.typing import ResponseReturnValue
from werkzeug.middleware.proxy_fix import ProxyFix

from db_viewer import URL_PREFIX, make_blueprint
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

REQUIRED_SETTINGS = (
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "DB_VIEWER_SECRET_KEY",
    "DB_VIEWER_ALLOWED_EMAILS",
)
GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"
SESSION_LIFETIME = timedelta(hours=12)
# Endpoints reachable without a session: the sign-in flow itself.
PUBLIC_ENDPOINTS = frozenset({"login", "auth_callback", "logout"})


def parse_allowed_emails(raw: str) -> FrozenSet[str]:
    return frozenset(email.lower() for email in raw.replace(",", " ").split())


def _is_viewer_path(path: str) -> bool:
    """Only ever send a signed-in user back into the viewer (no open redirect)."""
    return path.startswith(f"{URL_PREFIX}/")


def create_app(
    firestore_service: Any = None, env: Mapping[str, str] = os.environ
) -> Flask:
    """Build the app (served by viewer_wsgi.py)."""
    setup_logger()
    missing = [name for name in REQUIRED_SETTINGS if not env.get(name)]
    allowed = parse_allowed_emails(env.get("DB_VIEWER_ALLOWED_EMAILS", ""))
    if missing or not allowed:
        raise RuntimeError(
            f"Database viewer not configured; missing: {', '.join(missing)}"
        )

    if firestore_service is None:
        from services.firestore_service import FirestoreService

        firestore_service = FirestoreService()

    app = Flask(__name__)
    # Cloud Run terminates TLS, so the scheme comes from its X-Forwarded-Proto;
    # it makes the OAuth redirect URI https. A spoofed value only produces a
    # redirect URI Google doesn't have registered.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1)  # type: ignore[method-assign]
    app.config.update(
        SECRET_KEY=env["DB_VIEWER_SECRET_KEY"],
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=SESSION_LIFETIME,
    )

    oauth = OAuth(app)
    google = oauth.register(
        "google",
        client_id=env["GOOGLE_OAUTH_CLIENT_ID"],
        client_secret=env["GOOGLE_OAUTH_CLIENT_SECRET"],
        server_metadata_url=GOOGLE_METADATA_URL,
        client_kwargs={"scope": "openid email"},
    )

    @app.before_request
    def _require_sign_in() -> Optional[ResponseReturnValue]:
        if request.endpoint in PUBLIC_ENDPOINTS:
            return None
        email: Optional[str] = session.get("email")
        if email is None:
            if request.method == "GET":
                session["next"] = request.full_path
            return redirect(url_for("login"))
        if email not in allowed:
            # Signed in before being taken off the allowlist.
            session.clear()
            abort(403)
        return None

    @app.after_request
    def _harden(response: Response) -> Response:
        response.headers["X-Frame-Options"] = "DENY"
        # The viewer links out to the pages users analyzed; don't hand them its URLs.
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.route("/")
    def index() -> ResponseReturnValue:
        return redirect(f"{URL_PREFIX}/")

    @app.route("/auth/login")
    def login() -> ResponseReturnValue:
        response: Response = google.authorize_redirect(
            url_for("auth_callback", _external=True)
        )
        return response

    @app.route("/auth/callback")
    def auth_callback() -> ResponseReturnValue:
        try:
            # Checks state and the ID token's signature, audience, issuer,
            # expiry and nonce; "userinfo" holds its claims.
            token = google.authorize_access_token()
        except OAuthError as e:
            logger.warning("Database-viewer sign-in failed: %s", e)
            abort(400)
        claims = token.get("userinfo") or {}
        email = str(claims.get("email", "")).lower()
        if claims.get("email_verified") is not True or email not in allowed:
            logger.warning("Refusing database-viewer sign-in by %r", email)
            session.clear()
            abort(403)

        next_url = session.get("next", "")
        session.clear()
        session.permanent = True
        session["email"] = email
        logger.info("Database-viewer sign-in by %s", email)
        return redirect(next_url if _is_viewer_path(next_url) else f"{URL_PREFIX}/")

    @app.route("/auth/logout")
    def logout() -> str:
        session.clear()
        return "Signed out."

    app.register_blueprint(make_blueprint(firestore_service, local_only=False))
    return app
