"""Shared fixtures for the integration tests.

These tests run end-to-end against the Firestore emulator with the real
desktop-server worker and a running LM Studio. There is no Gemini fallback: if
the desktop-server fails, the tests fail (rather than silently using Gemini).

See tests/integration/run.sh for how to start the emulator + worker and run the
suite.
"""

import os
import sys
import time

import pytest
import requests

# Ensure the repo root is importable (services/, analysis_core.py, ...).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Project id used by desktop-server's emulator tooling.
PROJECT_ID = "desktop-server-test"


def _lms_reachable() -> bool:
    """Mirror desktop-server's reachability probe of the LM Studio server."""
    base = os.getenv("LMS_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    try:
        resp = requests.get(base + "/models", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def analysis_core():
    """An AnalysisCore wired to the emulator, desktop-only (no Gemini fallback)."""
    if not os.getenv("FIRESTORE_EMULATOR_HOST"):
        pytest.skip(
            "FIRESTORE_EMULATOR_HOST not set; run via tests/integration/run.sh"
        )
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
    os.environ["USE_DESKTOP_SERVER"] = "true"
    os.environ["DISABLE_GEMINI_FALLBACK"] = "true"
    os.environ.setdefault("LMS_MODEL", "openai/gpt-oss-20b")

    if not _lms_reachable():
        pytest.skip("LM Studio not reachable; start it with `lms server start`")

    from analysis_core import AnalysisCore

    core = AnalysisCore(enable_database=False)

    # Wait for the desktop-server worker heartbeat (written every ~10s).
    deadline = time.time() + 30
    while time.time() < deadline:
        if core.desktop_service.is_available():
            return core
        time.sleep(1)
    pytest.skip(
        "desktop-server worker heartbeat not detected; is the worker running "
        "against the emulator?"
    )
