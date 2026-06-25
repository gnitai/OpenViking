# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
Search Service for OpenViking.

Provides semantic search operations: search, find.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from openviking.core.uri_validation import validate_optional_viking_uris
from openviking.observability.events import try_publish_event
from openviking.server.identity import RequestContext
from openviking.storage.viking_fs import VikingFS
from openviking.utils.embedding_input import estimate_embedding_input_tokens
from openviking_cli.exceptions import InvalidArgumentError, NotInitializedError
from openviking_cli.utils import get_logger

if TYPE_CHECKING:
    from openviking.session import Session

logger = get_logger(__name__)


def _ensure_non_empty_query(query: str) -> None:
    if not query.strip():
        raise InvalidArgumentError("Search query must not be empty.")


def _result_token_count(result: Any) -> int:
    """Estimate the tokens in a FindResult's returned text payload.

    Counts the abstract + overview of every matched context, the text fields
    actually surfaced to the caller by find/search. Uses the local CJK-aware
    estimator (no real tokenizer exists in the codebase), so this is an estimate.
    """
    total = 0
    for bucket in (result.memories, result.resources, result.skills):
        for ctx in bucket:
            total += estimate_embedding_input_tokens(getattr(ctx, "abstract", "") or "")
            total += estimate_embedding_input_tokens(getattr(ctx, "overview", "") or "")
    return total


def _emit_retrieval_call(operation: str, result: Any, ctx: RequestContext) -> None:
    """Best-effort emit a `retrieval.call` event for per-account usage rollups.

    Identity is taken explicitly from `ctx` rather than ambient observability
    context, so per-account attribution does not depend on the root context being
    populated at this layer. Never raises: instrumentation must not break the
    search response.
    """
    try:
        result_count = getattr(result, "total", 0) or (
            len(result.memories) + len(result.resources) + len(result.skills)
        )
        try_publish_event(
            "retrieval.call",
            {
                "operation": operation,
                "account_id": ctx.account_id,
                "user_id": ctx.user.user_id,
                "agent_id": ctx.user.agent_id,
                "result_count": result_count,
                "result_token_count": _result_token_count(result),
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("failed to emit retrieval.call event", exc_info=True)


class SearchService:
    """Semantic search service."""

    def __init__(self, viking_fs: Optional[VikingFS] = None):
        self._viking_fs = viking_fs

    def set_viking_fs(self, viking_fs: VikingFS) -> None:
        """Set VikingFS instance (for deferred initialization)."""
        self._viking_fs = viking_fs

    def _ensure_initialized(self) -> VikingFS:
        """Ensure VikingFS is initialized."""
        if not self._viking_fs:
            raise NotInitializedError("VikingFS")
        return self._viking_fs

    async def search(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        session: Optional["Session"] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
    ) -> Any:
        """Complex search with session context.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            session: Session object for context
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult
        """
        _ensure_non_empty_query(query)
        target_uri = validate_optional_viking_uris(target_uri, field_name="target_uri")
        viking_fs = self._ensure_initialized()

        session_info = None
        if session:
            session_info = await session.get_context_for_search(query)

        result = await viking_fs.search(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            session_info=session_info,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
        )
        _emit_retrieval_call("search", result, ctx)
        return result

    async def find(
        self,
        query: str,
        ctx: RequestContext,
        target_uri: Union[str, List[str]] = "",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict] = None,
        level: Optional[List[int]] = None,
    ) -> Any:
        """Semantic search without session context.

        Args:
            query: Query string
            target_uri: Target directory URI(s), supports str or List[str]
            limit: Max results
            score_threshold: Score threshold
            filter: Metadata filters
            level: Filter by level (0=abstract, 1=overview, 2=file)

        Returns:
            FindResult
        """
        _ensure_non_empty_query(query)
        target_uri = validate_optional_viking_uris(target_uri, field_name="target_uri")
        viking_fs = self._ensure_initialized()
        result = await viking_fs.find(
            query=query,
            ctx=ctx,
            target_uri=target_uri,
            limit=limit,
            score_threshold=score_threshold,
            filter=filter,
            level=level,
        )
        _emit_retrieval_call("find", result, ctx)
        return result
