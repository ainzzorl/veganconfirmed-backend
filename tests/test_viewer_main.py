"""Tests for the deployed viewer's Google sign-in gate (viewer_main.py)."""

import pytest
from authlib.integrations.flask_client.apps import FlaskOAuth2App

from db_viewer import URL_PREFIX
from tests.test_db_viewer import DOCS, JOBS, FakeFirestoreService
from viewer_main import create_app

ENV = {
    "GOOGLE_OAUTH_CLIENT_ID": "client-id",
    "GOOGLE_OAUTH_CLIENT_SECRET": "client-secret",
    "DB_VIEWER_SECRET_KEY": "secret-key",
    "DB_VIEWER_ALLOWED_EMAILS": "Me@example.com, other@example.com",
}
BASE_URL = "https://viewer.example"
# Every request reaches Cloud Run through Google's front end, so it is never
# local in db_viewer's sense.
PROXIED = {"X-Forwarded-For": "203.0.113.7"}


@pytest.fixture
def client():
    app = create_app(FakeFirestoreService(DOCS, JOBS), env=ENV)
    return app.test_client()


def _get(client, path):
    return client.get(path, base_url=BASE_URL, headers=PROXIED)


def _sign_in_as(client, email):
    with client.session_transaction(base_url=BASE_URL) as session:
        session["email"] = email


@pytest.mark.parametrize(
    "override", [{"DB_VIEWER_ALLOWED_EMAILS": " , "}, {"DB_VIEWER_SECRET_KEY": ""}]
)
def test_refuses_to_start_unconfigured(override):
    with pytest.raises(RuntimeError):
        create_app(FakeFirestoreService(DOCS), env={**ENV, **override})


def test_signed_out_requests_go_to_sign_in(client):
    response = _get(client, f"{URL_PREFIX}/alpha?format=json")
    assert response.status_code == 302
    assert response.location.endswith("/auth/login")


def test_signed_in_user_sees_the_viewer(client):
    _sign_in_as(client, "me@example.com")
    response = _get(client, f"{URL_PREFIX}/")
    assert response.status_code == 200
    assert b"Leather sneakers" in response.data


def test_session_of_an_email_no_longer_allowed_is_refused(client):
    _sign_in_as(client, "former@example.com")
    assert _get(client, f"{URL_PREFIX}/").status_code == 403
    # ... and is dropped, so the next request starts over at sign-in.
    assert _get(client, f"{URL_PREFIX}/").status_code == 302


def _fake_token(monkeypatch, claims):
    monkeypatch.setattr(
        FlaskOAuth2App, "authorize_access_token", lambda self: {"userinfo": claims}
    )


@pytest.mark.parametrize(
    "next_path, expected",
    [
        (f"{URL_PREFIX}/?q=leather", f"{URL_PREFIX}/?q=leather"),
        ("//evil.example/", f"{URL_PREFIX}/"),
    ],
)
def test_sign_in_returns_to_the_viewer_only(client, monkeypatch, next_path, expected):
    with client.session_transaction(base_url=BASE_URL) as session:
        session["next"] = next_path
    _fake_token(monkeypatch, {"email": "ME@example.com", "email_verified": True})

    response = _get(client, "/auth/callback")
    assert response.status_code == 302
    assert response.location == expected
    assert _get(client, f"{URL_PREFIX}/").status_code == 200


@pytest.mark.parametrize(
    "claims",
    [
        {"email": "stranger@example.com", "email_verified": True},
        {"email": "me@example.com", "email_verified": False},
        {},
    ],
)
def test_sign_in_refuses_anyone_else(client, monkeypatch, claims):
    _fake_token(monkeypatch, claims)
    assert _get(client, "/auth/callback").status_code == 403
    assert _get(client, f"{URL_PREFIX}/").status_code == 302
