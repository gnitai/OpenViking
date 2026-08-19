# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-user attribution of LLM-proxy traffic.

The W LLM proxy meters chat traffic per user through its JWT gateway, but
embeddings reach LiteLLM directly under a shared service key -- so those spend
logs carry no end user unless the request body says who it is for. These tests
cover the ``metadata.user_id`` the proxy's pre-call callback reads to attribute
them, and the boundary that keeps it off non-W providers.
"""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from openviking.models.embedder import OpenAIDenseEmbedder
from openviking.models.llm_credentials import (
    apply_llm_credentials,
    apply_llm_request_metadata,
    bind_llm_credentials,
    bind_llm_user_id,
    get_llm_credentials,
    reset_llm_credentials,
)
from openviking.models.vlm.backends.openai_vlm import OpenAIVLM
from openviking.storage.queuefs.embedding_msg import EmbeddingMsg


@pytest.fixture
def bound_user():
    """Bind a W credential for the duration of one test."""
    token = bind_llm_credentials("jwt-abc", "project-7", "3122")
    try:
        yield
    finally:
        reset_llm_credentials(token)


def _jwt(payload: dict) -> str:
    """Build an unsigned token shaped like the gateway's, for claim reading."""
    segment = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{segment}.signature"


class TestUserIdFromAuthToken:
    """The gateway bills by the token's claim, so attribution reads it too."""

    def test_claim_wins_over_the_identity_header(self):
        """Indexing arrives under a shared service identity, not a person.

        Trusting the header would file every project's vector spend under one
        name, while the same run's gateway calls land on the real user.
        """
        token = bind_llm_credentials(_jwt({"user_id": 3122}), "project-7", "shared")
        try:
            creds = get_llm_credentials()
            assert creds.user_id == "3122"
        finally:
            reset_llm_credentials(token)

    def test_identity_header_is_the_fallback(self):
        """A non-JWT credential (local runs, test keys) still attributes."""
        token = bind_llm_credentials("not-a-jwt", "project-7", "3122")
        try:
            assert get_llm_credentials().user_id == "3122"
        finally:
            reset_llm_credentials(token)

    def test_token_without_the_claim_falls_back(self):
        token = bind_llm_credentials(_jwt({"exp": 1}), "project-7", "3122")
        try:
            assert get_llm_credentials().user_id == "3122"
        finally:
            reset_llm_credentials(token)

    def test_undecodable_token_never_raises(self):
        """A malformed token must not break the request it rides on."""
        for bad in ("a.b.c", "..", _jwt({}) + ".extra", "!!!.!!!.!!!"):
            token = bind_llm_credentials(bad, "project-7", "3122")
            try:
                assert get_llm_credentials().user_id == "3122"
            finally:
                reset_llm_credentials(token)

    def test_anonymous_claim_is_not_an_identity(self):
        token = bind_llm_credentials(_jwt({"user_id": "default"}), "project-7", "3122")
        try:
            assert get_llm_credentials().user_id == "3122"
        finally:
            reset_llm_credentials(token)


class TestApplyLLMUserMetadata:
    """The attribution-only injector used by service-key routes."""

    def test_no_op_without_bound_credentials(self):
        kwargs = {}
        apply_llm_request_metadata(kwargs)
        assert kwargs == {}

    def test_no_op_when_credential_carries_no_user_id(self):
        token = bind_llm_credentials("jwt-abc", "project-7")
        try:
            kwargs = {}
            apply_llm_request_metadata(kwargs)
            assert kwargs == {}
        finally:
            reset_llm_credentials(token)

    def test_injects_user_id_into_extra_body_metadata(self, bound_user):
        kwargs = {}
        apply_llm_request_metadata(kwargs)
        assert kwargs == {"extra_body": {"metadata": {"user_id": "3122"}}}

    def test_feature_tag_is_opt_in(self, bound_user):
        """Callers that have no feature to declare must not invent an empty list."""
        kwargs = {}
        apply_llm_request_metadata(kwargs)
        assert "feature_tags" not in kwargs["extra_body"]["metadata"]

    def test_feature_tag_rides_alongside_the_user(self, bound_user):
        kwargs = {}
        apply_llm_request_metadata(kwargs, feature_tag="feature:x")
        assert kwargs["extra_body"]["metadata"] == {
            "user_id": "3122",
            "feature_tags": ["feature:x"],
        }

    def test_feature_tag_merges_without_duplicating(self, bound_user):
        kwargs = {"extra_body": {"metadata": {"feature_tags": ["feature:x"]}}}
        apply_llm_request_metadata(kwargs, feature_tag="feature:x")
        assert kwargs["extra_body"]["metadata"]["feature_tags"] == ["feature:x"]

    def test_feature_tag_appends_to_existing_tags(self, bound_user):
        kwargs = {"extra_body": {"metadata": {"feature_tags": ["feature:x"]}}}
        apply_llm_request_metadata(kwargs, feature_tag="feature:y")
        assert kwargs["extra_body"]["metadata"]["feature_tags"] == ["feature:x", "feature:y"]

    def test_feature_tag_needs_a_bound_credential(self):
        """Off W there is no proxy to read the tag; do not leak it to a provider."""
        kwargs = {}
        apply_llm_request_metadata(kwargs, feature_tag="feature:x")
        assert kwargs == {}

    def test_never_touches_auth(self, bound_user):
        """Embedding routes authenticate with the proxy's own key, not the JWT."""
        kwargs = {}
        apply_llm_request_metadata(kwargs)
        assert "extra_headers" not in kwargs

    def test_preserves_existing_extra_body_and_metadata(self, bound_user):
        kwargs = {"extra_body": {"input_type": "query", "metadata": {"tenant": "t1"}}}
        apply_llm_request_metadata(kwargs)
        assert kwargs["extra_body"] == {
            "input_type": "query",
            "metadata": {"tenant": "t1", "user_id": "3122"},
        }

    def test_does_not_mutate_the_caller_s_nested_dicts(self, bound_user):
        """The embedder rebuilds extra_body per call; shared dicts must not leak."""
        shared_metadata = {"tenant": "t1"}
        kwargs = {"extra_body": {"metadata": shared_metadata}}
        apply_llm_request_metadata(kwargs)
        assert shared_metadata == {"tenant": "t1"}

    def test_coerces_non_string_user_id(self):
        token = bind_llm_credentials("jwt-abc", "project-7", 3122)
        try:
            kwargs = {}
            apply_llm_request_metadata(kwargs)
            assert kwargs["extra_body"]["metadata"]["user_id"] == "3122"
        finally:
            reset_llm_credentials(token)


class TestApplyLLMCredentialsCarriesUserId:
    """The JWT path attributes too, so gateway and direct routes agree."""

    def test_credentials_include_user_metadata(self, bound_user):
        kwargs = {}
        apply_llm_credentials(kwargs)
        assert kwargs["extra_headers"]["Authorization"] == "Bearer jwt-abc"
        assert kwargs["extra_body"] == {
            "project_id": "project-7",
            "metadata": {"user_id": "3122"},
        }

    def test_user_id_is_optional(self):
        token = bind_llm_credentials("jwt-abc", "project-7")
        try:
            kwargs = {}
            apply_llm_credentials(kwargs)
            assert kwargs["extra_body"] == {"project_id": "project-7"}
        finally:
            reset_llm_credentials(token)

    def test_anonymous_sentinel_binds_as_no_user(self):
        """A ROOT reindex names an account but no user; "default" is not an identity.

        Binding it would pool every such request's spend under one fabricated end
        user, which is worse than the unattributed row we would otherwise get.
        """
        token = bind_llm_credentials("jwt-abc", "project-7", "default")
        try:
            kwargs = {}
            apply_llm_credentials(kwargs)
            assert "metadata" not in kwargs["extra_body"]
        finally:
            reset_llm_credentials(token)

    def test_binding_requires_an_auth_token(self):
        """A user id alone does not mark a request as W-routed."""
        assert bind_llm_credentials(None, "project-7", "3122") is None
        kwargs = {}
        apply_llm_request_metadata(kwargs)
        assert kwargs == {}


class TestBindLLMUserId:
    """Attribution-only binding for workers that hold no JWT."""

    def test_binds_user_without_auth_token(self):
        token = bind_llm_user_id("3122")
        try:
            kwargs = {}
            apply_llm_request_metadata(kwargs)
            assert kwargs["extra_body"]["metadata"] == {"user_id": "3122"}
        finally:
            reset_llm_credentials(token)

    def test_does_not_override_client_auth(self):
        """Without a JWT there is nothing better to authenticate with."""
        token = bind_llm_user_id("3122")
        try:
            kwargs = {}
            apply_llm_credentials(kwargs, {"X-Title": "ov"})
            assert "extra_headers" not in kwargs
            assert kwargs["extra_body"] == {"metadata": {"user_id": "3122"}}
        finally:
            reset_llm_credentials(token)

    def test_falsy_or_anonymous_user_id_binds_nothing(self):
        assert bind_llm_user_id(None) is None
        assert bind_llm_user_id("") is None
        assert bind_llm_user_id("default") is None
        kwargs = {}
        apply_llm_request_metadata(kwargs)
        assert kwargs == {}


class TestEmbeddingMsgCarriesUser:
    """Indexing embeddings run on the queue, so the id rides on the message."""

    def test_defaults_from_the_bound_credential(self, bound_user):
        msg = EmbeddingMsg(message="text", context_data={"uri": "wfs://x"})
        assert msg.llm_user_id == "3122"

    def test_defaults_to_none_without_a_credential(self):
        msg = EmbeddingMsg(message="text", context_data={"uri": "wfs://x"})
        assert msg.llm_user_id is None

    def test_explicit_value_wins_over_the_contextvar(self, bound_user):
        msg = EmbeddingMsg(
            message="text",
            context_data={"uri": "wfs://x"},
            llm_user_id="explicit",
        )
        assert msg.llm_user_id == "explicit"

    def test_survives_the_durable_queue_round_trip(self, bound_user):
        msg = EmbeddingMsg(message="text", context_data={"uri": "wfs://x"})
        restored = EmbeddingMsg.from_json(msg.to_json())
        assert restored.llm_user_id == "3122"

    def test_round_trip_of_a_message_written_before_this_field_existed(self):
        """Queued messages outlive deploys; an older payload must still load."""
        legacy = {"message": "text", "context_data": {}, "telemetry_id": "t1"}
        assert EmbeddingMsg.from_dict(legacy).llm_user_id is None

    def test_legacy_payload_does_not_inherit_the_deserializing_task_s_user(self, bound_user):
        """Deserializing is not a claim of ownership.

        If from_dict fell through to the contextvar, a worker that happens to be
        acting for one user would silently take over another user's queued vector.
        """
        legacy = {"message": "text", "context_data": {}, "telemetry_id": "t1"}
        assert EmbeddingMsg.from_dict(legacy).llm_user_id is None


def _make_mock_embedding_client():
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 8)],
        usage=None,
    )
    return client


class TestEmbedderAttribution:
    """Every embedding call says who it is for and what it is for."""

    @patch("openviking.models.embedder.openai_embedders.openai.OpenAI")
    def test_embed_attaches_user_and_feature_tag(self, mock_openai_class, bound_user):
        mock_client = _make_mock_embedding_client()
        mock_openai_class.return_value = mock_client

        embedder = OpenAIDenseEmbedder(
            model_name="voyage/voyage-code-3",
            api_key="sk-test",
            api_base="https://llm.example.com/v1",
            dimension=8,
        )
        embedder.embed("hello")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["extra_body"]["metadata"] == {
            "user_id": "3122",
            "feature_tags": ["feature:context-indexing:embedding"],
        }

    @patch("openviking.models.embedder.openai_embedders.openai.OpenAI")
    def test_batch_embed_attaches_user_and_feature_tag(self, mock_openai_class, bound_user):
        mock_client = _make_mock_embedding_client()
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 8), MagicMock(embedding=[0.2] * 8)],
            usage=None,
        )
        mock_openai_class.return_value = mock_client

        embedder = OpenAIDenseEmbedder(
            model_name="voyage/voyage-code-3",
            api_key="sk-test",
            api_base="https://llm.example.com/v1",
            dimension=8,
        )
        embedder.embed_batch(["a", "b"])

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["extra_body"]["metadata"] == {
            "user_id": "3122",
            "feature_tags": ["feature:context-indexing:embedding"],
        }

    @patch("openviking.models.embedder.openai_embedders.openai.OpenAI")
    def test_reindex_without_a_user_is_still_labelled(self, mock_openai_class):
        """A ROOT reindex names no user, but its vector spend is still indexing."""
        token = bind_llm_credentials("jwt-abc", "project-7", "default")
        try:
            mock_client = _make_mock_embedding_client()
            mock_openai_class.return_value = mock_client

            embedder = OpenAIDenseEmbedder(
                model_name="voyage/voyage-code-3",
                api_key="sk-test",
                api_base="https://llm.example.com/v1",
                dimension=8,
            )
            embedder.embed("hello")

            call_kwargs = mock_client.embeddings.create.call_args.kwargs
            assert call_kwargs["extra_body"]["metadata"] == {
                "feature_tags": ["feature:context-indexing:embedding"],
            }
        finally:
            reset_llm_credentials(token)

    @patch("openviking.models.embedder.openai_embedders.openai.OpenAI")
    def test_first_party_openai_never_receives_metadata(self, mock_openai_class, bound_user):
        """A credential says the request is W-routed, not that this endpoint is.

        Chat can route through the W proxy while embeddings point straight at
        OpenAI, which rejects unknown body fields outright -- that would fail
        every embed and query-embed call, not just lose attribution.
        """
        mock_client = _make_mock_embedding_client()
        mock_openai_class.return_value = mock_client

        embedder = OpenAIDenseEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
            dimension=8,
        )
        embedder.embed("hello")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    @patch("openviking.models.embedder.openai_embedders.openai.OpenAI")
    def test_default_openai_endpoint_never_receives_metadata(self, mock_openai_class, bound_user):
        """No api_base means the SDK's own default: first-party OpenAI."""
        mock_client = _make_mock_embedding_client()
        mock_openai_class.return_value = mock_client

        embedder = OpenAIDenseEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            dimension=8,
        )
        embedder.embed("hello")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    @patch("openviking.models.embedder.openai_embedders.openai.AzureOpenAI")
    def test_azure_never_receives_metadata(self, mock_azure_class, bound_user):
        mock_client = _make_mock_embedding_client()
        mock_azure_class.return_value = mock_client

        embedder = OpenAIDenseEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            api_base="https://example-resource.openai.azure.com",
            provider="azure",
            dimension=8,
        )
        embedder.embed("hello")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    @patch("openviking.models.embedder.openai_embedders.openai.OpenAI")
    def test_no_credential_sends_no_metadata(self, mock_openai_class):
        """A stock OpenAI deployment must not receive a W metadata field."""
        mock_client = _make_mock_embedding_client()
        mock_openai_class.return_value = mock_client

        embedder = OpenAIDenseEmbedder(
            model_name="text-embedding-3-small",
            api_key="sk-test",
            dimension=8,
        )
        embedder.embed("hello")

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert "extra_body" not in call_kwargs

    @patch("openviking.models.embedder.openai_embedders.openai.OpenAI")
    def test_user_metadata_coexists_with_input_type(self, mock_openai_class, bound_user):
        mock_client = _make_mock_embedding_client()
        mock_openai_class.return_value = mock_client

        embedder = OpenAIDenseEmbedder(
            model_name="bge-m3",
            api_key="sk-test",
            api_base="https://llm.example.com/v1",
            dimension=8,
            query_param="query",
        )
        embedder.embed("hello", is_query=True)

        call_kwargs = mock_client.embeddings.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "input_type": "query",
            "metadata": {
                "user_id": "3122",
                "feature_tags": ["feature:context-indexing:embedding"],
            },
        }


class TestVLMAttribution:
    """The VLM path carries the same identifier alongside its JWT override."""

    @patch("openviking.models.vlm.backends.openai_vlm.openai.OpenAI")
    def test_completion_attaches_user_metadata(self, mock_openai_class, bound_user):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"), finish_reason="stop")],
            usage=None,
        )
        mock_openai_class.return_value = mock_client

        vlm = OpenAIVLM(
            {
                "api_key": "sk-test",
                "api_base": "https://llm.example.com/user",
                "model": "indexing",
            }
        )
        vlm.get_completion("hello")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["metadata"] == {"user_id": "3122"}
        assert call_kwargs["extra_body"]["project_id"] == "project-7"
        assert call_kwargs["extra_headers"]["Authorization"] == "Bearer jwt-abc"
