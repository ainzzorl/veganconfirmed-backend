import logging
from datetime import datetime, timezone
from typing import List, Optional
from firebase_admin import credentials, firestore, initialize_app
from pydantic import BaseModel
from google.cloud.firestore import Query
from models.database_model import (
    APICallRecord,
    FeedbackRecord,
    MenuRecord,
    STATUS_OK,
    ShoppingItemRecord,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Cap on the page text stored alongside a record. This is a storage concern, not
# an analysis one — it keeps a document well inside Firestore's 1MB limit — so
# it is deliberately separate from the model's page budget
# (page_prompt.PAGE_CONTENT_TOKEN_LIMIT) and stays counted in characters.
STORED_CONTENT_CHAR_LIMIT = 20000


def _token_usage_doc(usage: Optional[TokenUsage]) -> Optional[dict]:
    """Render token usage as a Firestore map (``None`` when unreported)."""
    return usage.model_dump() if usage else None


def _token_usage_from_doc(data: Optional[dict]) -> Optional[TokenUsage]:
    """Read the token-usage map back, tolerating records written before it."""
    return TokenUsage(**data) if isinstance(data, dict) else None


def _branch_doc(branch: Optional[BaseModel]) -> Optional[dict]:
    """Render a per-kind analysis branch as a Firestore map (``None`` if unset)."""
    return branch.model_dump() if branch else None


def _shopping_item_from_doc(data: Optional[dict]) -> Optional[ShoppingItemRecord]:
    """Read the product branch back; absent on menus, "other" pages and failures."""
    return ShoppingItemRecord(**data) if isinstance(data, dict) else None


def _menu_from_doc(data: Optional[dict]) -> Optional[MenuRecord]:
    """Read the menu branch back; absent on products, "other" pages and failures."""
    return MenuRecord(**data) if isinstance(data, dict) else None


def _feedback_from_doc(data: Optional[dict]) -> Optional[FeedbackRecord]:
    """Read the feedback map back; absent on every analysis nobody rated."""
    return FeedbackRecord(**data) if isinstance(data, dict) else None


class FirestoreService:
    def __init__(self, collection_name: str = "api_calls"):
        """Initialize Firestore service"""
        self.collection_name = collection_name

        # Initialize Firebase Admin SDK
        try:
            # Check if Firebase app is already initialized
            try:
                self.db = firestore.client()
                logger.info("Firestore service initialized with existing Firebase app")
            except ValueError:
                # Firebase app not initialized, initialize it
                logger.info("Initializing Firebase Admin SDK...")
                initialize_app()
                self.db = firestore.client()
                logger.info("Firestore service initialized with new Firebase app")
        except Exception as e:
            logger.warning(f"Failed to initialize with default credentials: {e}")
            # For local development, you might need to set GOOGLE_APPLICATION_CREDENTIALS
            # or use a service account key file
            raise Exception(
                "Firestore initialization failed. Please ensure proper credentials are configured."
            )

    def save_api_call(self, record: APICallRecord) -> str:
        """
        Save an API call record to Firestore

        Args:
            record: APICallRecord object containing the call data

        Returns:
            str: The document ID of the inserted record
        """
        try:
            # Prepare document data
            doc_data = {
                "ip_hash": record.ip_hash,
                "country": record.country,
                "user_agent": record.user_agent,
                "content": record.content[:STORED_CONTENT_CHAR_LIMIT],
                "item_url": record.item_url,
                "origin_url": record.origin_url,
                "title": record.title,
                "page_kind": record.page_kind,
                # The kind-specific halves, one map each. The branch that does
                # not apply is written as null rather than omitted, so every
                # document carries both fields in the same shape.
                "shopping_item": _branch_doc(record.shopping_item),
                "menu": _branch_doc(record.menu),
                "language": record.language,
                "summary": record.summary,
                # How the request ended. A failed request is stored too, with
                # the analysis fields empty and ``error`` saying what went
                # wrong; see analysis_core._save_failure.
                "status": record.status,
                "error": record.error,
                "user_avoided_ingredients": record.user_avoided_ingredients,
                "service": record.service,
                "model": record.model,
                "token_usage": _token_usage_doc(record.token_usage),
                # Desktop-server job this analysis ran as, so the record can be
                # followed back to the request/response it exchanged with
                # LM Studio. Null for anything Gemini served.
                "lms_job_id": record.lms_job_id,
                # How long the call took, as its two ends; see APICallRecord.
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "extension_version": record.extension_version,
                "installation_id": record.installation_id,
                "created_at": datetime.now(timezone.utc),
                "trigger_type": record.trigger_type,
                "trigger_element_text": record.trigger_element_text,
                "trigger_element_selector": record.trigger_element_selector,
                # What the page declared about itself, the scope rule that
                # read it, and the page kinds that scope left the model to
                # choose from; see APICallRecord.
                "page_signals": record.page_signals,
                "page_scope_rule": record.page_scope_rule,
                "page_scope_kinds": record.page_scope_kinds,
            }

            # Add document to Firestore
            doc_ref = self.db.collection(self.collection_name).add(doc_data)
            document_id = doc_ref[1].id

            logger.info(f"Saved API call record with ID: {document_id}")
            return document_id

        except Exception as e:
            logger.error(f"Failed to save API call record: {e}")
            raise

    def save_feedback(
        self, analysis_id: str, rating: str, comment: Optional[str]
    ) -> bool:
        """Attach a user's verdict to the analysis it is about.

        Returns False when there is no such analysis, which is how the endpoint
        tells a stale or invented ID from a real one.

        A rating is sent the moment a thumb is clicked and a comment, if one is
        written at all, arrives in a second call; both are the same merged
        write, so the second updates the first rather than recording a second
        opinion. ``created_at`` is carried over from the earlier write so it
        goes on meaning when the thumb was clicked, and a comment already
        stored is kept when a later call carries none — switching the rating
        should not silently erase what the user wrote.
        """
        try:
            doc_ref = self.db.collection(self.collection_name).document(analysis_id)
            snapshot = doc_ref.get()
            if not snapshot.exists:
                return False

            now = datetime.now(timezone.utc)
            existing = (snapshot.to_dict() or {}).get("feedback") or {}
            doc_ref.set(
                {
                    "feedback": {
                        "rating": rating,
                        "comment": (
                            comment if comment is not None else existing.get("comment")
                        ),
                        "created_at": existing.get("created_at") or now,
                        "updated_at": now,
                    }
                },
                merge=True,
            )

            logger.info(f"Saved '{rating}' feedback on {analysis_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save feedback on {analysis_id}: {e}")
            raise

    def get_api_calls(
        self,
        limit: int = 100,
        offset: int = 0,
        is_vegan: Optional[bool] = None,
    ) -> List[APICallRecord]:
        """
        Retrieve API call records from Firestore

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            is_vegan: Filter by vegan status

        Returns:
            List of APICallRecord objects
        """
        try:
            query = self.db.collection(self.collection_name)

            # Apply filters
            if is_vegan is not None:
                query = query.where("shopping_item.is_vegan", "==", is_vegan)

            # Order by created_at descending
            query = query.order_by("created_at", direction=Query.DESCENDING)

            # Apply pagination
            if offset > 0:
                # For offset, we need to get documents and skip
                docs = query.limit(offset + limit).stream()
                records = []
                for i, doc in enumerate(docs):
                    if i >= offset:
                        record = self._doc_to_record(doc)
                        records.append(record)
                        if len(records) >= limit:
                            break
                return records
            else:
                # No offset, just limit
                docs = query.limit(limit).stream()
                records = []
                for doc in docs:
                    record = self._doc_to_record(doc)
                    records.append(record)
                return records

        except Exception as e:
            logger.error(f"Failed to retrieve API call records: {e}")
            raise

    def _doc_to_record(self, doc) -> APICallRecord:
        """Convert Firestore document to APICallRecord"""
        data = doc.to_dict()
        return APICallRecord(
            id=doc.id,
            ip_hash=data.get("ip_hash"),
            country=data.get("country"),
            user_agent=data.get("user_agent"),
            content=data.get("content", ""),
            item_url=data.get("item_url", ""),
            origin_url=data.get("origin_url", ""),
            title=data.get("title", ""),
            page_kind=data.get("page_kind"),
            shopping_item=_shopping_item_from_doc(data.get("shopping_item")),
            menu=_menu_from_doc(data.get("menu")),
            language=data.get("language"),
            summary=data.get("summary", ""),
            # Records written before failures were stored carry no status, and
            # every one of them is a request that was answered.
            status=data.get("status") or STATUS_OK,
            error=data.get("error"),
            user_avoided_ingredients=data.get("user_avoided_ingredients"),
            service=data.get("service"),
            model=data.get("model"),
            token_usage=_token_usage_from_doc(data.get("token_usage")),
            lms_job_id=data.get("lms_job_id"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            extension_version=data.get("extension_version"),
            installation_id=data.get("installation_id"),
            created_at=data.get("created_at"),
            trigger_type=data.get("trigger_type"),
            trigger_element_text=data.get("trigger_element_text"),
            trigger_element_selector=data.get("trigger_element_selector"),
            page_signals=data.get("page_signals"),
            page_scope_rule=data.get("page_scope_rule"),
            page_scope_kinds=data.get("page_scope_kinds"),
            feedback=_feedback_from_doc(data.get("feedback")),
        )

    def get_raw_documents(self, limit: int = 500) -> List[dict]:
        """Read the most recent documents as plain dicts, newest first.

        Unlike :meth:`get_api_calls` this keeps every field the document
        actually holds rather than the subset ``APICallRecord`` declares, so
        the debugging viewer shows fields written by newer (or older) code as
        they are. The document ID is merged in under ``id``.
        """
        try:
            docs = (
                self.db.collection(self.collection_name)
                .order_by("created_at", direction=Query.DESCENDING)
                .limit(limit)
                .stream()
            )
            return [{**(doc.to_dict() or {}), "id": doc.id} for doc in docs]

        except Exception as e:
            logger.error(f"Failed to read raw documents: {e}")
            raise

    def get_raw_document(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> Optional[dict]:
        """Read one document as a plain dict, or ``None`` if it is missing.

        ``collection_name`` defaults to this service's own collection; the
        viewer passes another one to follow a record's link into a collection
        this service does not write, such as the desktop-server's ``lms_jobs``.
        """
        collection = collection_name or self.collection_name
        try:
            doc = self.db.collection(collection).document(document_id).get()
            if not doc.exists:
                return None
            return {**(doc.to_dict() or {}), "id": doc.id}

        except Exception as e:
            logger.error(f"Failed to read document {collection}/{document_id}: {e}")
            raise

    def get_statistics(self) -> dict:
        """
        Get database statistics

        Returns:
            Dictionary containing various statistics
        """
        try:
            # Get total records
            total_records = len(list(self.db.collection(self.collection_name).stream()))

            # Get records by vegan status
            vegan_records = (
                self.db.collection(self.collection_name)
                .where("shopping_item.is_vegan", "==", True)
                .stream()
            )
            non_vegan_records = (
                self.db.collection(self.collection_name)
                .where("shopping_item.is_vegan", "==", False)
                .stream()
            )

            vegan_count = len(list(vegan_records))
            non_vegan_count = len(list(non_vegan_records))

            # Get confidence statistics
            confidence_stats = {}
            docs = self.db.collection(self.collection_name).stream()
            for doc in docs:
                data = doc.to_dict()
                # Only a product page has one; menus, "other" pages and failures
                # all fall in the "unknown" bucket.
                shopping_item = data.get("shopping_item") or {}
                confidence = shopping_item.get("confidence_level") or "unknown"
                confidence_stats[confidence] = confidence_stats.get(confidence, 0) + 1

            # Get recent activity (last 24 hours)
            from datetime import timedelta

            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            recent_docs = (
                self.db.collection(self.collection_name)
                .where("created_at", ">=", yesterday)
                .stream()
            )
            recent_count = len(list(recent_docs))

            return {
                "total_records": total_records,
                "vegan_stats": {True: vegan_count, False: non_vegan_count},
                "confidence_stats": confidence_stats,
                "recent_activity_24h": recent_count,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise

    def cleanup_old_records(self, days: int = 30) -> int:
        """
        Clean up old records older than specified days

        Args:
            days: Number of days to keep records

        Returns:
            int: Number of records deleted
        """
        try:
            from datetime import timedelta

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            # Get old documents
            old_docs = (
                self.db.collection(self.collection_name)
                .where("created_at", "<", cutoff_date)
                .stream()
            )

            deleted_count = 0
            for doc in old_docs:
                doc.reference.delete()
                deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            raise
