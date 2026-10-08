"""Tests for the feedback a user leaves on an analysis.

Feedback is stored as a map on the analysis's own ``api_calls`` document, and
arrives in up to two calls: the thumb the moment it is clicked, then a comment
if the user writes one. Both ends are driven through fakes, so these run
without the emulator or any credentials.
"""

from datetime import datetime, timezone

import pytest
from flask import Flask

from analysis_core import AnalysisCore
from services.firestore_service import FirestoreService

ANALYSIS_ID = "doc-1"
EARLIER = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


class FakeDocRef:
    """A document that can be read back and merged into."""

    def __init__(self, store, doc_id):
        self.store = store
        self.doc_id = doc_id

    def get(self):
        return FakeSnapshot(self.store.get(self.doc_id))

    def set(self, data, merge=False):
        assert merge, "feedback must merge, or it would wipe the analysis"
        self.store.setdefault(self.doc_id, {}).update(data)


class FakeSnapshot:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return self._data


class FakeCollection:
    def __init__(self, store):
        self.store = store

    def document(self, doc_id):
        return FakeDocRef(self.store, doc_id)


class FakeDb:
    def __init__(self, store):
        self.store = store

    def collection(self, name):
        assert name == "api_calls"
        return FakeCollection(self.store)


@pytest.fixture
def store():
    """One stored analysis, as if it had just been answered."""
    return {ANALYSIS_ID: {"item_url": "https://example.com/soap", "summary": "..."}}


@pytest.fixture
def firestore_service(store):
    service = FirestoreService.__new__(FirestoreService)
    service.collection_name = "api_calls"
    service.db = FakeDb(store)
    return service


@pytest.fixture
def core(firestore_service):
    core = AnalysisCore.__new__(AnalysisCore)
    core.database_service = firestore_service
    return core


def feedback_of(store):
    return store[ANALYSIS_ID]["feedback"]


# --- the write ---------------------------------------------------------------


def test_thumb_is_stored_on_the_analysis(core, store):
    body, status = core.record_feedback({"analysis_id": ANALYSIS_ID, "rating": "down"})

    assert (body, status) == ({"status": "ok"}, 200)
    assert feedback_of(store)["rating"] == "down"
    assert feedback_of(store)["comment"] is None
    # Merged in, not written over: the analysis is still there.
    assert store[ANALYSIS_ID]["summary"] == "..."


def test_comment_merges_onto_the_rating_already_sent(core, store):
    core.record_feedback({"analysis_id": ANALYSIS_ID, "rating": "down"})
    first_seen = feedback_of(store)["created_at"]

    core.record_feedback(
        {
            "analysis_id": ANALYSIS_ID,
            "rating": "down",
            "comment": "  The soap is dove, it has tallow.  ",
        }
    )

    stored = feedback_of(store)
    assert stored["comment"] == "The soap is dove, it has tallow."
    # One opinion, not two: created_at still says when the thumb was clicked.
    assert stored["created_at"] == first_seen
    assert stored["updated_at"] >= first_seen


def test_switching_the_rating_keeps_the_comment(core, store):
    core.record_feedback(
        {"analysis_id": ANALYSIS_ID, "rating": "down", "comment": "Wrong."}
    )
    core.record_feedback({"analysis_id": ANALYSIS_ID, "rating": "up"})

    stored = feedback_of(store)
    assert stored["rating"] == "up"
    assert stored["comment"] == "Wrong."


# --- what it refuses ---------------------------------------------------------


def test_unknown_analysis_is_not_found(core, store):
    body, status = core.record_feedback({"analysis_id": "nope", "rating": "up"})

    assert status == 404
    assert "nope" not in store


def test_rating_must_be_a_thumb(core, store):
    _, status = core.record_feedback({"analysis_id": ANALYSIS_ID, "rating": "meh"})

    assert status == 400
    assert "feedback" not in store[ANALYSIS_ID]


def test_overlong_comment_is_refused(core, store):
    _, status = core.record_feedback(
        {"analysis_id": ANALYSIS_ID, "rating": "down", "comment": "x" * 2001}
    )

    assert status == 400
    assert "feedback" not in store[ANALYSIS_ID]


# --- the endpoint ------------------------------------------------------------


@pytest.fixture
def client(core):
    """The Flask route over the fake core, registered as app.py registers it."""
    import app as app_module

    flask_app = Flask(__name__)
    flask_app.add_url_rule(
        "/api/feedback", view_func=app_module.record_feedback, methods=["POST"]
    )
    app_module.analysis_core = core
    return flask_app.test_client()


def test_endpoint_stores_feedback(client, store):
    response = client.post(
        "/api/feedback", json={"analysis_id": ANALYSIS_ID, "rating": "up"}
    )

    assert response.status_code == 200
    assert feedback_of(store)["rating"] == "up"


def test_endpoint_rejects_an_empty_body(client):
    assert client.post("/api/feedback", json={}).status_code == 400
