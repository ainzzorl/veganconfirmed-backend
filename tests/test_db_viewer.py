"""Tests for the local-only database viewer.

The viewer is driven through a fake FirestoreService, so these run without the
emulator or any credentials.
"""

from datetime import datetime, timedelta, timezone

import pytest
from flask import Flask

from db_viewer import URL_PREFIX, register_db_viewer


class FakeFirestoreService:
    """Stands in for FirestoreService, serving fixed collections of documents."""

    collection_name = "api_calls"

    def __init__(self, docs, jobs=None):
        self.docs = docs
        # Keyed by collection, mirroring the second collection the real service
        # is asked for: the desktop-server's jobs.
        self.jobs = jobs or {}

    def get_raw_documents(self, limit: int = 500):
        return self.docs[:limit]

    def get_raw_document(self, document_id: str, collection_name: str = None):
        if collection_name and collection_name != self.collection_name:
            return self.jobs.get(document_id)
        return next((doc for doc in self.docs if doc["id"] == document_id), None)


# The two ends of a call, which the viewer shows as the time it took.
STARTED_AT = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


# A completed desktop-server job, in the shape its worker writes: the
# OpenAI-format request that went to LM Studio and the response that came back.
JOBS = {
    "job-1": {
        "id": "job-1",
        "status": "success",
        "model": "openai/gpt-oss-120b",
        "created_at": None,
        "updated_at": None,
        "request": {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "system", "content": "You are an analyst advising a vegan."},
                {"role": "user", "content": "URL: https://shop.example/sneakers"},
            ],
            "max_tokens": 8192,
            "temperature": 0.4,
        },
        "response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "reasoning": "The upper is described as leather.",
                        "content": '{"is_vegan": false}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 900, "total_tokens": 1000},
        },
    }
}


DOCS = [
    {
        "id": "alpha",
        "created_at": None,
        "item_url": "https://shop.example/sneakers",
        "origin_url": "https://shop.example/sneakers",
        "title": "Leather sneakers",
        "content": "Upper: full grain leather.",
        "page_kind": "shopping_item",
        "shopping_item": {
            "is_vegan": False,
            "is_cruelty_free": None,
            "cruelty_free_explanation": None,
            "confidence_level": "high",
        },
        "menu": None,
        "summary": "Made of leather; the upper is full grain leather.",
        "service": "desktop",
        "model": "openai/gpt-oss-120b",
        "token_usage": {"prompt_tokens": 900, "total_tokens": 1000},
        "lms_job_id": "job-1",
        "page_scope_kinds": ["shopping_item", "other"],
        "page_scope_rule": "declared_item",
        "started_at": STARTED_AT,
        "finished_at": STARTED_AT + timedelta(milliseconds=4321),
        "installation_id": "install-1",
    },
    {
        "id": "beta",
        "created_at": None,
        "item_url": "https://maps.example/cafe-verde",
        "title": "Cafe Verde",
        "content": "Falafel wrap. Mushroom risotto.",
        "page_kind": "restaurant_menu",
        "shopping_item": None,
        "summary": "Several vegan mains on a mixed menu.",
        "service": "gemini",
        "model": "gemini-2.0-flash",
        "token_usage": {"total_tokens": 2000},
        "page_scope_kinds": ["restaurant_menu", "other"],
        "page_scope_rule": "menu_source",
        "started_at": STARTED_AT,
        "finished_at": STARTED_AT + timedelta(milliseconds=850),
        "installation_id": "install-2",
        "menu": {
            "restaurant_name": "Cafe Verde",
            "vegan_friendliness": "medium",
            "items": [
            {
                "name": "Falafel wrap",
                "section": "Mains",
                "verdict": "vegan",
                "reason": "Chickpea patty in flatbread with tahini.",
                "explicit_ingredients": ["chickpea patty", "flatbread", "tahini"],
                "inferred_ingredients": [],
                "user_avoided_ingredients": [],
            },
            {
                "name": "Mushroom risotto",
                "section": "Mains",
                "verdict": "not_vegan",
                "reason": "Contains parmesan and butter.",
                "explicit_ingredients": ["mushroom", "parmesan"],
                "inferred_ingredients": ["butter", "rice"],
                "user_avoided_ingredients": [],
            },
            ],
        },
    },
    # A request that produced no analysis: stored with the reason instead.
    {
        "id": "delta",
        "created_at": None,
        "item_url": "https://shop.example/bamboo-socks",
        "title": "Bamboo socks",
        "content": "Bamboo viscose socks.",
        "status": "analysis_failed",
        "error": "RuntimeError: desktop-server response missing choices",
        "installation_id": "install-1",
    },
]


@pytest.fixture
def client():
    app = Flask(__name__)
    assert register_db_viewer(app, FakeFirestoreService(DOCS, JOBS))
    return app.test_client()


def test_list_shows_every_record(client):
    body = client.get(f"{URL_PREFIX}/").get_data(as_text=True)
    assert "Leather sneakers" in body
    assert "Cafe Verde" in body
    assert "not vegan" in body
    assert "openai/gpt-oss-120b" in body


def test_filter_by_page_kind(client):
    body = client.get(f"{URL_PREFIX}/?page_kind=restaurant_menu").get_data(as_text=True)
    assert "Cafe Verde" in body
    assert "Leather sneakers" not in body


def test_the_scope_a_call_ran_under_is_shown_and_filterable(client):
    """The stored page kinds read as the one-word scope the logs use.

    The document holds the kinds themselves, so both the column and the filter
    go through the same label — a list rendered raw would make the filter's
    options unusable.
    """
    body = client.get(f"{URL_PREFIX}/?page_scope_kinds=menu_or_other").get_data(
        as_text=True
    )
    assert "Cafe Verde" in body
    assert "Leather sneakers" not in body

    detail = client.get(f"{URL_PREFIX}/alpha").get_data(as_text=True)
    assert "shopping_item, other" in detail
    assert "declared_item" in detail

    # "delta" was written before the scope was stored; it must not show up as
    # a scope of its own in the filter's options.
    options = client.get(f"{URL_PREFIX}/").get_data(as_text=True)
    assert 'value="item_or_other"' in options
    assert 'value="None"' not in options


def test_search_looks_inside_stored_content(client):
    body = client.get(f"{URL_PREFIX}/?q=falafel").get_data(as_text=True)
    assert "Cafe Verde" in body
    assert "Leather sneakers" not in body


def test_list_json(client):
    payload = client.get(
        f"{URL_PREFIX}/?format=json&page_kind=shopping_item"
    ).get_json()
    assert payload["matching"] == 1
    assert payload["records"][0]["id"] == "alpha"


def test_paging_splits_the_matches(client):
    first = client.get(f"{URL_PREFIX}/?page_size=1&format=json").get_json()
    second = client.get(f"{URL_PREFIX}/?page_size=1&offset=1&format=json").get_json()
    assert [r["id"] for r in first["records"]] == ["alpha"]
    assert [r["id"] for r in second["records"]] == ["beta"]


def test_detail_shows_the_stored_content(client):
    body = client.get(f"{URL_PREFIX}/alpha").get_data(as_text=True)
    assert "Upper: full grain leather." in body
    assert "Made of leather; the upper is full grain leather." in body
    # Every stored field is reachable, including ones no group lists.
    assert "install-1" in body


def test_list_counts_the_dishes_of_a_menu(client):
    body = client.get(f"{URL_PREFIX}/?page_kind=restaurant_menu").get_data(as_text=True)
    assert "2 dishes" in body


def test_detail_shows_every_dish(client):
    body = client.get(f"{URL_PREFIX}/beta").get_data(as_text=True)
    assert "Falafel wrap" in body
    assert "Mushroom risotto" in body
    assert "Contains parmesan and butter." in body
    # Both ingredient lists get a column of their own.
    assert "mushroom, parmesan" in body
    assert "butter, rice" in body
    # Verdicts are shown as prose (the raw document below still has the enum
    # spelling), and tallied above the table.
    assert '<td class="v-not">not vegan</td>' in body
    assert '<td class="v-vegan">vegan</td>' in body
    assert "vegan: 1" in body
    # No dish names an avoid-list ingredient, so that column stays off.
    assert "Avoided" not in body


def test_detail_shows_the_avoid_list_column_when_a_dish_names_one(client):
    docs = [dict(DOCS[1], id="gamma")]
    docs[0]["menu"] = dict(
        DOCS[1]["menu"],
        items=[
            dict(DOCS[1]["menu"]["items"][0], user_avoided_ingredients=["palm oil"])
        ],
    )
    app = Flask(__name__)
    register_db_viewer(app, FakeFirestoreService(docs))

    body = app.test_client().get(f"{URL_PREFIX}/gamma").get_data(as_text=True)
    assert "Avoided" in body
    assert "palm oil" in body


def test_detail_of_a_record_without_dishes_has_no_dish_table(client):
    body = client.get(f"{URL_PREFIX}/alpha").get_data(as_text=True)
    assert "Menu dishes" not in body


def test_detail_shows_the_linked_desktop_job(client):
    body = client.get(f"{URL_PREFIX}/alpha").get_data(as_text=True)
    assert "lms_jobs/job-1" in body
    # The prompt as it was actually sent, and what came back.
    assert "You are an analyst advising a vegan." in body
    assert "The upper is described as leather." in body
    assert "stop" in body


def test_detail_of_a_gemini_record_has_no_job_section(client):
    # Gemini runs no job, so the field is empty and the section is absent —
    # only the record's own provider row mentions one.
    body = client.get(f"{URL_PREFIX}/beta").get_data(as_text=True)
    assert "lms_jobs/" not in body
    assert "Raw job document" not in body


def test_a_record_whose_job_is_gone_still_renders(client):
    docs = [dict(DOCS[0], id="gamma", lms_job_id="job-missing")]
    app = Flask(__name__)
    register_db_viewer(app, FakeFirestoreService(docs, JOBS))

    body = app.test_client().get(f"{URL_PREFIX}/gamma").get_data(as_text=True)
    assert "Leather sneakers" in body
    assert "lms_jobs/job-missing is gone" in body


def test_detail_json_includes_the_job(client):
    payload = client.get(f"{URL_PREFIX}/alpha?format=json").get_json()
    assert payload["lms_job"]["status"] == "success"


def test_the_list_shows_how_long_each_call_took(client):
    body = client.get(f"{URL_PREFIX}/").get_data(as_text=True)
    # Sub-second calls in milliseconds, longer ones in seconds, and the mean
    # over the records that carry a duration at all.
    assert "4.3 s" in body
    assert "850 ms" in body
    assert "mean: 2.6 s" in body


def test_a_record_written_before_calls_were_timed_still_renders(client):
    body = client.get(f"{URL_PREFIX}/delta").get_data(as_text=True)
    assert "Bamboo socks" in body
    assert "Took" in body


def test_search_finds_a_record_by_its_job_id(client):
    body = client.get(f"{URL_PREFIX}/?q=job-1").get_data(as_text=True)
    assert "Leather sneakers" in body
    assert "Cafe Verde" not in body


def test_detail_json(client):
    assert (
        client.get(f"{URL_PREFIX}/beta?format=json").get_json()["title"] == "Cafe Verde"
    )


def test_list_marks_a_failed_request(client):
    body = client.get(f"{URL_PREFIX}/").get_data(as_text=True)
    assert "Bamboo socks" in body
    assert "failed" in body
    # The error stands in for the summary the record does not have.
    assert "desktop-server response missing choices" in body


def test_filter_by_status(client):
    failures = client.get(f"{URL_PREFIX}/?status=analysis_failed").get_data(
        as_text=True
    )
    assert "Bamboo socks" in failures
    assert "Leather sneakers" not in failures

    # Records written before failures were stored carry no status, and must
    # still show up as the successes they are.
    successes = client.get(f"{URL_PREFIX}/?status=ok").get_data(as_text=True)
    assert "Leather sneakers" in successes
    assert "Bamboo socks" not in successes


def test_detail_shows_why_a_request_failed(client):
    body = client.get(f"{URL_PREFIX}/delta").get_data(as_text=True)
    assert "analysis_failed" in body
    assert "RuntimeError: desktop-server response missing choices" in body
    # The page that failed is still there to look at.
    assert "Bamboo viscose socks." in body


def test_search_finds_a_record_by_its_error(client):
    body = client.get(f"{URL_PREFIX}/?q=missing choices").get_data(as_text=True)
    assert "Bamboo socks" in body
    assert "Leather sneakers" not in body


def test_missing_record_is_404(client):
    assert client.get(f"{URL_PREFIX}/nope").status_code == 404


def test_page_content_is_escaped():
    docs = [dict(DOCS[0], id="evil", title="<script>alert(1)</script>")]
    app = Flask(__name__)
    register_db_viewer(app, FakeFirestoreService(docs))
    body = app.test_client().get(f"{URL_PREFIX}/").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


@pytest.mark.parametrize(
    "kwargs",
    [
        {"headers": {"X-Forwarded-For": "203.0.113.7"}},
        {"headers": {"X-Real-IP": "203.0.113.7"}},
        {"environ_base": {"REMOTE_ADDR": "203.0.113.7"}},
    ],
)
def test_non_local_requests_are_hidden(client, kwargs):
    """A proxied or remote request must not see the viewer at all."""
    assert client.get(f"{URL_PREFIX}/", **kwargs).status_code == 404


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("ENABLE_DB_VIEWER", "false")
    app = Flask(__name__)
    assert register_db_viewer(app, FakeFirestoreService(DOCS)) is False
    assert app.test_client().get(f"{URL_PREFIX}/").status_code == 404


def test_not_registered_without_a_database():
    app = Flask(__name__)
    assert register_db_viewer(app, None) is False
