"""Tests for what an analysis actually stores in the ``api_calls`` collection.

Both sides of the write are driven through fakes, so these run without the
emulator or any credentials: ``AnalysisCore._save_record`` builds the record,
and ``FirestoreService`` turns it into the document.
"""

from datetime import datetime, timedelta, timezone

import pytest

from analysis_core import AnalysisCore
from models.analysis_model import PageAnalysis
from models.database_model import (
    APICallRecord,
    MenuRecord,
    ProviderCall,
    ShoppingItemRecord,
    STATUS_ANALYSIS_FAILED,
    STATUS_INVALID_REQUEST,
    STATUS_OK,
)
from services.firestore_service import FirestoreService
from services.privacy import IpPrivacy
from services.page_scope import ALL_PAGE_KINDS, MENU_OR_OTHER, ScopeDecision

MENU_ITEMS = [
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
]


class FakeDocRef:
    id = "doc-1"


class FakeCollection:
    def __init__(self, written):
        self.written = written

    def add(self, doc_data):
        self.written.append(doc_data)
        # Mirrors the (timestamp, DocumentReference) tuple Firestore returns.
        return (None, FakeDocRef())


class FakeDb:
    """Stands in for the Firestore client, keeping what was written."""

    def __init__(self):
        self.written = []

    def collection(self, name):
        assert name == "api_calls"
        return FakeCollection(self.written)


DESKTOP_CALL = ProviderCall(
    service="desktop",
    model="openai/gpt-oss-20b",
    lms_job_id="job-1",
)


class FakeDoc:
    """A read-side document snapshot."""

    def __init__(self, data, doc_id="doc-1"):
        self.id = doc_id
        self._data = data

    def to_dict(self):
        return self._data


@pytest.fixture
def firestore_service():
    """A FirestoreService over a fake client, bypassing its Firebase setup."""
    service = FirestoreService.__new__(FirestoreService)
    service.collection_name = "api_calls"
    service.db = FakeDb()
    return service


def _record(**overrides) -> APICallRecord:
    fields = {
        "ip_hash": "0123456789abcdef",
        "content": "The Olive Branch — Menu.",
        "item_url": "https://example.com/olive-branch/menu",
        "origin_url": "https://example.com/olive-branch/menu",
        "title": "The Olive Branch",
        "page_kind": "restaurant_menu",
        "summary": "Several vegan mains.",
    }
    fields.update(overrides)
    return APICallRecord(**fields)


def test_the_menu_branch_is_written(firestore_service):
    firestore_service.save_api_call(
        _record(
            menu=MenuRecord(
                restaurant_name="The Olive Branch",
                vegan_friendliness="medium",
                items=MENU_ITEMS,
            )
        )
    )

    (doc,) = firestore_service.db.written
    assert doc["menu"]["items"] == MENU_ITEMS
    assert doc["menu"]["restaurant_name"] == "The Olive Branch"
    assert doc["menu"]["vegan_friendliness"] == "medium"
    assert doc["shopping_item"] is None


def test_a_product_writes_the_other_branch(firestore_service):
    firestore_service.save_api_call(
        _record(
            page_kind="shopping_item",
            shopping_item=ShoppingItemRecord(is_vegan=False, confidence_level="high"),
        )
    )

    (doc,) = firestore_service.db.written
    assert doc["shopping_item"] == {
        "explicit_ingredients": [],
        "inferred_ingredients": [],
        "animal_derived_ingredients": [],
        "is_vegan": False,
        "is_cruelty_free": None,
        "cruelty_free_explanation": None,
        "confidence_level": "high",
    }
    # Both keys are always present, so every document has one shape.
    assert doc["menu"] is None


def test_the_desktop_job_link_is_written(firestore_service):
    firestore_service.save_api_call(_record(lms_job_id="job-1"))

    (doc,) = firestore_service.db.written
    assert doc["lms_job_id"] == "job-1"


def test_a_gemini_call_links_to_no_job(firestore_service):
    firestore_service.save_api_call(_record(service="gemini"))

    (doc,) = firestore_service.db.written
    assert doc["lms_job_id"] is None


def test_the_desktop_job_link_is_read_back(firestore_service):
    record = firestore_service._doc_to_record(FakeDoc({"lms_job_id": "job-1"}))

    assert record.lms_job_id == "job-1"


def test_the_branches_are_read_back(firestore_service):
    record = firestore_service._doc_to_record(
        FakeDoc({"menu": {"restaurant_name": "The Olive Branch", "items": MENU_ITEMS}})
    )

    assert record.menu.items == MENU_ITEMS
    assert record.menu.restaurant_name == "The Olive Branch"
    assert record.shopping_item is None


def test_a_document_without_branches_reads_back_as_none(firestore_service):
    """Failures store no analysis, and neither branch is written for them."""
    record = firestore_service._doc_to_record(FakeDoc({"page_kind": "other"}))

    assert record.shopping_item is None
    assert record.menu is None


class FakeDatabaseService:
    """Captures the record ``_save_record`` builds."""

    def __init__(self):
        self.records = []

    def save_api_call(self, record):
        self.records.append(record)
        return "doc-1"


def _bare_core(database_service=None) -> AnalysisCore:
    """An AnalysisCore over a fake database, without providers."""
    core = AnalysisCore.__new__(AnalysisCore)
    core.enable_database = True
    core.database_service = database_service or FakeDatabaseService()
    core.ip_privacy = IpPrivacy(secret="test-secret", geoip_db="/nonexistent.mmdb")
    return core


class FakeRequest:
    """The subset of PageAnalysisRequest that ``_save_record`` reads."""

    content = "The Olive Branch — Menu."
    url = "https://example.com/olive-branch/menu"
    title = "The Olive Branch"
    language = "en"
    extension_version = "1.0.4"
    installation_id = "install-1"
    trigger_type = "manual"
    trigger_element_text = None
    trigger_element_selector = None
    page_signals = {"og_type": "article", "schema_types": ["Menu", "MenuItem"]}


def test_save_record_carries_the_dish_verdicts():
    core = _bare_core()

    core._save_record(
        analysis_request=FakeRequest(),
        analysis=PageAnalysis.from_flat(
            {
                "page_kind": "restaurant_menu",
                "items": MENU_ITEMS,
                "restaurant_name": "The Olive Branch",
                "vegan_friendliness": "medium",
                "summary": "Several vegan mains.",
            }
        ),
        client_ip="127.0.0.1",
        user_agent=None,
        provider_call=DESKTOP_CALL,
    )

    (record,) = core.database_service.records
    assert record.menu.items == MENU_ITEMS
    assert record.menu.restaurant_name == "The Olive Branch"
    assert record.shopping_item is None


def test_save_record_carries_what_the_page_declared():
    """The raw declaration and the rule that read it, stored on every record.

    Both halves are needed to measure a pruning rule: once a rule fires, the
    branch it dropped is not one the model could have answered with, so its
    precision can only be checked by re-deriving it offline from the signals of
    requests it did not narrow.
    """
    core = _bare_core()

    core._save_record(
        analysis_request=FakeRequest(),
        analysis=PageAnalysis.from_flat({"page_kind": "restaurant_menu"}),
        client_ip="127.0.0.1",
        user_agent=None,
        provider_call=DESKTOP_CALL,
        scope=ScopeDecision(MENU_OR_OTHER, "declared_menu"),
    )

    (record,) = core.database_service.records
    assert record.page_signals == FakeRequest.page_signals
    assert record.page_scope_rule == "declared_menu"
    assert record.page_scope_kinds == list(MENU_OR_OTHER)


def test_save_record_carries_the_desktop_job_id():
    # The job the analysis ran as is only known to the provider that served it;
    # _save_record is where it lands on the record.
    core = _bare_core()

    core._save_record(
        analysis_request=FakeRequest(),
        analysis=PageAnalysis.from_flat({"page_kind": "shopping_item"}),
        client_ip="127.0.0.1",
        user_agent=None,
        provider_call=DESKTOP_CALL,
    )

    (record,) = core.database_service.records
    assert record.lms_job_id == "job-1"
    assert record.service == "desktop"
    assert record.model == "openai/gpt-oss-20b"


def _save_a_record(database_service):
    core = _bare_core(database_service)
    return core._save_record(
        analysis_request=FakeRequest(),
        analysis=PageAnalysis.from_flat({"page_kind": "shopping_item"}),
        client_ip="127.0.0.1",
        user_agent=None,
        provider_call=DESKTOP_CALL,
    )


def test_save_record_returns_the_id_the_response_carries():
    """The document ID goes back to the extension as ``analysis_id``.

    It is the only handle feedback has on an analysis, so losing it here would
    leave every answer unratable.
    """
    assert _save_a_record(FakeDatabaseService()) == "doc-1"


def test_a_failed_save_costs_the_id_and_nothing_else():
    class BrokenDatabase:
        def save_api_call(self, record):
            raise RuntimeError("Firestore is unreachable")

    # No exception escapes: the caller is on its way to answering the user,
    # and only the feedback control is lost.
    assert _save_a_record(BrokenDatabase()) is None


def test_a_successful_call_is_stored_as_ok(firestore_service):
    firestore_service.save_api_call(_record())

    (doc,) = firestore_service.db.written
    assert doc["status"] == STATUS_OK
    assert doc["error"] is None


def test_a_failure_is_stored_with_its_reason(firestore_service):
    firestore_service.save_api_call(
        _record(
            summary="",
            confidence="",
            status=STATUS_ANALYSIS_FAILED,
            error="RuntimeError: desktop-server response missing choices",
        )
    )

    (doc,) = firestore_service.db.written
    assert doc["status"] == STATUS_ANALYSIS_FAILED
    assert doc["error"] == "RuntimeError: desktop-server response missing choices"


def test_a_record_written_before_failures_were_stored_reads_as_ok(firestore_service):
    record = firestore_service._doc_to_record(FakeDoc({"summary": "Vegan."}))

    assert record.status == STATUS_OK
    assert record.error is None


def test_the_failure_status_is_read_back(firestore_service):
    record = firestore_service._doc_to_record(
        FakeDoc({"status": STATUS_INVALID_REQUEST, "error": "ValidationError: nope"})
    )

    assert record.status == STATUS_INVALID_REQUEST
    assert record.error == "ValidationError: nope"


VALID_PAYLOAD = {
    "url": "https://example.com/olive-branch/menu",
    "content": "The Olive Branch — Menu.",
    "timestamp": "2024-01-01T12:00:00Z",
    "title": "The Olive Branch",
    "installation_id": "install-1",
}


def _core(analysis_error: Exception = None) -> AnalysisCore:
    """An AnalysisCore whose analysis fails, over a fake database."""
    core = _bare_core()

    def fail(analysis_request):
        raise analysis_error

    core._run_analysis = fail
    return core


def test_a_call_records_how_long_it_took():
    core = _bare_core()
    core._run_analysis = lambda analysis_request: (
        {"page_kind": "shopping_item", "is_vegan": True},
        DESKTOP_CALL,
        ScopeDecision(ALL_PAGE_KINDS, "none"),
    )

    _, status_code = core.analyze_page(VALID_PAYLOAD, client_ip="127.0.0.1")

    assert status_code == 200
    (record,) = core.database_service.records
    assert record.started_at is not None
    assert record.finished_at >= record.started_at


def test_the_response_names_the_record_it_was_stored_as():
    """``analysis_id`` is the extension's only handle for rating an answer."""
    core = _bare_core()
    core._run_analysis = lambda analysis_request: (
        {"page_kind": "shopping_item", "is_vegan": True},
        DESKTOP_CALL,
        ScopeDecision(ALL_PAGE_KINDS, "none"),
    )

    response, status_code = core.analyze_page(VALID_PAYLOAD, client_ip="127.0.0.1")

    assert status_code == 200
    assert response["analysis_id"] == "doc-1"


def test_a_malformed_page_signal_does_not_fail_the_call():
    """The hint is advisory and client-supplied; a bad one is not a bad request.

    Rejecting it would turn a page that could still have been analyzed into a
    400, so the field is dropped and the analysis goes ahead without it.
    """
    core = _bare_core()
    core._run_analysis = lambda analysis_request: (
        {"page_kind": "restaurant_menu"},
        DESKTOP_CALL,
        ScopeDecision(ALL_PAGE_KINDS, "none"),
    )

    _, status_code = core.analyze_page(
        {**VALID_PAYLOAD, "page_signals": "Menu"}, client_ip="127.0.0.1"
    )

    assert status_code == 200
    (record,) = core.database_service.records
    assert record.page_signals is None


def test_a_failure_records_how_long_it_took():
    # A request that fell over took time too, and is worth comparing against
    # the ones that worked.
    core = _core(RuntimeError("boom"))

    core.analyze_page(VALID_PAYLOAD, client_ip="127.0.0.1")

    (record,) = core.database_service.records
    assert record.started_at is not None
    assert record.finished_at >= record.started_at


def test_the_call_timestamps_are_written(firestore_service):
    started = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    firestore_service.save_api_call(
        _record(started_at=started, finished_at=started + timedelta(seconds=4))
    )

    (doc,) = firestore_service.db.written
    assert doc["started_at"] == started
    assert (doc["finished_at"] - doc["started_at"]).total_seconds() == 4


def test_an_analysis_failure_is_stored():
    core = _core(RuntimeError("desktop-server response missing choices"))

    response, status_code = core.analyze_page(VALID_PAYLOAD, client_ip="127.0.0.1")

    assert status_code == 500
    (record,) = core.database_service.records
    assert record.status == STATUS_ANALYSIS_FAILED
    assert record.error == "RuntimeError: desktop-server response missing choices"
    # The request itself is stored in full, so the page that failed can be
    # looked at (and re-run) afterwards.
    assert record.item_url == VALID_PAYLOAD["url"]
    assert record.content == VALID_PAYLOAD["content"]
    assert record.installation_id == "install-1"
    # A failure has no analysis to report.
    assert record.summary == ""
    # A failure has no analysis, so neither per-kind branch is filled.
    assert record.shopping_item is None
    assert record.menu is None


def test_an_invalid_request_is_stored():
    core = _core()
    payload = {"url": "https://example.com/socks", "title": "Socks"}  # no content

    response, status_code = core.analyze_page(payload, client_ip="127.0.0.1")

    assert status_code == 400
    (record,) = core.database_service.records
    assert record.status == STATUS_INVALID_REQUEST
    assert "content" in record.error
    assert record.item_url == "https://example.com/socks"
    assert record.title == "Socks"


def test_a_payload_of_the_wrong_shape_is_still_stored():
    # The whole point of the invalid-request record is payloads like this one,
    # so building it must not depend on any field having the right type.
    core = _core()

    _, status_code = core.analyze_page(
        {"url": 7, "content": None, "user_avoided_ingredients": "palm oil"},
        client_ip="127.0.0.1",
    )

    assert status_code == 400
    (record,) = core.database_service.records
    assert record.status == STATUS_INVALID_REQUEST
    assert record.item_url == "7"
    assert record.content == ""
    assert record.user_avoided_ingredients is None


def test_nothing_is_stored_without_a_database():
    core = _core(RuntimeError("boom"))
    core.enable_database = False
    core.database_service = None

    _, status_code = core.analyze_page(VALID_PAYLOAD, client_ip="127.0.0.1")

    assert status_code == 500


def test_the_client_ip_is_kept_only_as_a_hash():
    core = _bare_core()
    core._run_analysis = lambda analysis_request: (
        {"page_kind": "shopping_item", "is_vegan": True},
        DESKTOP_CALL,
        ScopeDecision(ALL_PAGE_KINDS, "none"),
    )

    core.analyze_page(VALID_PAYLOAD, client_ip="198.51.100.23")
    core.analyze_page({"url": "https://example.com"}, client_ip="198.51.100.23")

    answered, invalid = core.database_service.records
    assert answered.ip_hash and answered.ip_hash == invalid.ip_hash
    for record in (answered, invalid):
        assert "198.51.100.23" not in str(record.model_dump())


def test_the_url_loses_its_query_and_fragment_everywhere_but_the_response():
    url = "https://example.com/socks?session=abc123#reviews"
    seen = []
    core = _bare_core()

    def analyze(analysis_request):
        seen.append(analysis_request.url)
        return (
            {"page_kind": "shopping_item", "is_vegan": True},
            DESKTOP_CALL,
            ScopeDecision(ALL_PAGE_KINDS, "none"),
        )

    core._run_analysis = analyze

    response, _ = core.analyze_page({**VALID_PAYLOAD, "url": url})
    core.analyze_page({"url": url})  # invalid

    assert seen == ["https://example.com/socks"]
    assert [r.item_url for r in core.database_service.records] == [
        "https://example.com/socks"
    ] * 2
    assert response["url"] == url
