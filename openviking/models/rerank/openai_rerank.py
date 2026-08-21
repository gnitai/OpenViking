# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""
OpenAI-compatible Rerank API Client.

Supports third-party rerank services like Alibaba Cloud DashScope (qwen3-rerank)
via api_key + api_base configuration.
"""

# For logging, use Python's built-in logging
from typing import Dict, List, Optional

import requests

from openviking.models.embedder.openai_embedders import _normalize_host, _w_proxy_hosts
from openviking.models.llm_credentials import apply_llm_request_metadata
from openviking.models.rerank.base import RerankBase
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

# Feature label for rerank spend at the W proxy. A sub-tag of the retrieval
# feature rather than the indexing one beside it: reranking happens while
# answering a search, not while building the index, and pooling the two would
# hide which half of the pipeline the spend came from.
RERANK_FEATURE_TAG = "feature:context-retrieval:rerank"


class OpenAIRerankClient(RerankBase):
    """
    OpenAI-compatible rerank API client using Bearer token auth.

    Compatible with services like Alibaba Cloud DashScope.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model_name: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize OpenAI-compatible rerank client.

        Args:
            api_key: Bearer token for authentication
            api_base: Full endpoint URL for the rerank API
            model_name: Model name to use for reranking
            extra_headers: Optional extra headers for API requests
        """
        super().__init__()
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.extra_headers = extra_headers or {}
        self.provider = "openai"

    def rerank_batch(self, query: str, documents: List[str]) -> Optional[List[float]]:
        """
        Batch rerank documents against a query.

        Args:
            query: Query text
            documents: List of document texts to rank

        Returns:
            List of rerank scores for each document (same order as input),
            or None when rerank fails and the caller should fall back
        """
        if not documents:
            return []

        req_body = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }
        metadata = self._request_metadata()
        if metadata:
            req_body["metadata"] = metadata

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.extra_headers:
                headers.update(self.extra_headers)

            response = requests.post(
                url=self.api_base,
                headers=headers,
                json=req_body,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Update token usage tracking (estimate, OpenAI rerank doesn't provide token info)
            self._extract_and_update_token_usage(result, query, documents)

            # Standard OpenAI/Cohere rerank format: results[].{index, relevance_score}
            results = result.get("results")
            if not results:
                logger.warning(f"[OpenAIRerankClient] Unexpected response format: {result}")
                return None

            if len(results) != len(documents):
                logger.warning(
                    "[OpenAIRerankClient] Unexpected rerank result length: expected=%s actual=%s",
                    len(documents),
                    len(results),
                )
                return None

            # Results may not be in original order — sort by index
            scores = [0.0] * len(documents)
            for item in results:
                idx = item.get("index")
                if idx is None or not (0 <= idx < len(documents)):
                    logger.warning(
                        "[OpenAIRerankClient] Out-of-bounds or missing index in result: %s", item
                    )
                    return None
                scores[idx] = item.get("relevance_score", 0.0)

            logger.debug(f"[OpenAIRerankClient] Reranked {len(documents)} documents")
            return scores

        except Exception as e:
            logger.error(f"[OpenAIRerankClient] Rerank failed: {e}")
            return None

    def _is_w_proxy(self) -> bool:
        """True when this client points at the W LLM proxy.

        Only that proxy reads the feature tag, so only it is sent one. A rerank
        service that is not it would receive an unknown body field bought for
        nothing -- and some reject unknown fields outright.
        """
        return _normalize_host(self.api_base or "") in _w_proxy_hosts()

    def _request_metadata(self) -> Dict[str, object]:
        """W attribution for this call: who it is for, and what it is for.

        Rerank runs during the search request but reaches the proxy under the
        deployment's own service key rather than the caller's JWT, and it skips
        the gateway that would otherwise label it -- so the proxy can infer
        neither. Without this the spend lands as an unattributed, untagged row
        and is invisible in any per-user or per-feature view.

        Built through :func:`apply_llm_request_metadata`, then lifted out of the
        OpenAI-SDK ``extra_body`` envelope that helper writes into: this client
        posts a raw JSON body, and ``metadata`` belongs at its top level, which
        is where the proxy reads it.

        Nothing is sent anywhere but the W proxy. A bound credential says the
        *request* is W-routed; it says nothing about where *this client* points,
        and a deployment can route chat through the proxy while reranking against
        a vendor directly. Rerank bodies are vendor-specific schemas -- Voyage,
        Cohere, DashScope -- and an unknown field is rejected outright by some of
        them, which here would fail the call, get swallowed by the caller's
        fallback, and silently cost ranking quality. Since no vendor reads this
        field anyway, the destination alone decides, and an unset
        ``OPENVIKING_LLM_PROXY_HOSTS`` means nothing is volunteered.

        This is stricter than the embedder's equivalent gate, which allows any
        OpenAI-compatible gateway because embedding bodies are one shared schema.
        """
        if not self._is_w_proxy():
            return {}

        carrier: Dict[str, object] = {}
        apply_llm_request_metadata(
            carrier,
            feature_tag=RERANK_FEATURE_TAG,
            # A search may name no user -- an unauthenticated probe, an internal
            # health query -- and that spend is still ours to account for. Label
            # it against the W proxy even with no user to name.
            tag_without_credentials=True,
        )
        extra_body = carrier.get("extra_body") or {}
        return extra_body.get("metadata") or {}

    @classmethod
    def from_config(cls, config) -> Optional["OpenAIRerankClient"]:
        """
        Create OpenAIRerankClient from RerankConfig.

        Args:
            config: RerankConfig instance with provider='openai'

        Returns:
            OpenAIRerankClient instance or None if config is not available
        """
        if not config or not config.is_available():
            return None
        return cls(
            api_key=config.api_key,
            api_base=config.api_base,
            model_name=config.model or "qwen3-rerank",
            extra_headers=config.extra_headers,
        )
