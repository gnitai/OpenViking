# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-request LLM-proxy credentials for the W gateway.

W routes paid LLM traffic , whose gateway
authenticates the caller via a per-user JWT (``Authorization: Bearer <jwt>``)
and reads an optional ``project_id`` from the request body to deduct credits
from that user's plan.

VLM (L0/L1) generation does not run during the HTTP request -- it is performed
later by the semantic-queue worker. So the requesting user's JWT must be carried
from the request, through the durable queue message, into the deferred VLM call.
This module holds that credential in a contextvar that is bound at both entry
points (the request dependency and the queue worker) and read at the point where
OpenAI-SDK call kwargs are assembled.

When no credential is bound (local/non-W runs) every helper is a no-op, so
the VLM falls back to the static ``api_key`` configured for the backend.
"""

from __future__ import annotations

import base64
import binascii
import contextvars
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

#: Body key the W LLM proxy reads for per-user attribution. Its LiteLLM
#: pre-call callback copies ``metadata.user_id`` onto the request's end-user
#: (customer) field, so spend logs for traffic sent under a shared service key
#: are still attributable to the user it was done for.
_USER_METADATA_KEY = "user_id"

#: Body key the W gateway uses to pass feature tags through to the proxy's
#: callback, which merges them into the spend log's request tags. Traffic that
#: reaches the proxy directly has to write this key itself.
_FEATURE_TAGS_METADATA_KEY = "feature_tags"

#: OpenViking's stand-in for "this request named no user" -- a ROOT reindex, for
#: instance, only has to name an account. It is a sentinel, not an identity, so it
#: must never reach the proxy: an end user literally named "default" would pool
#: unrelated users' spend under one fabricated row.
_ANONYMOUS_USER_ID = "default"


def _real_user_id(user_id: Optional[str]) -> Optional[str]:
    """Return the user id, or None if it is absent or the anonymous sentinel."""
    if not user_id or user_id == _ANONYMOUS_USER_ID:
        return None
    return str(user_id)


@dataclass(frozen=True)
class LLMCredentials:
    """Caller credentials forwarded to the W LLM gateway.

    ``auth_token`` is absent on the attribution-only binding used by the
    embedding worker (see :func:`bind_llm_user_id`), which knows who the work is
    for but has no JWT to authenticate as.
    """

    auth_token: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None


_LLM_CREDENTIALS: contextvars.ContextVar[Optional[LLMCredentials]] = contextvars.ContextVar(
    "openviking_llm_credentials",
    default=None,
)


def _user_id_from_auth_token(token: Optional[str]) -> Optional[str]:
    """Read the ``user_id`` claim out of a W gateway JWT, if the token is one.

    The signature is deliberately not verified, and this value must not be
    trusted for anything beyond a label: the gateway verifies the same token on
    the routes that bill against it, so a forged one fails there regardless.

    What it buys is agreement. The gateway attributes its own traffic by this
    exact claim, so reading it here puts direct-to-LiteLLM rows (embeddings) on
    the same id as gateway rows for one indexing run -- rather than on whatever
    the caller happened to put in the identity header, which for indexing is a
    shared service identity rather than a person.
    """
    if not token or token.count(".") != 2:
        return None
    payload_segment = token.split(".")[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_segment + padding))
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _real_user_id(payload.get("user_id"))


def bind_llm_credentials(
    auth_token: Optional[str],
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[contextvars.Token]:
    """Bind credentials for the current task; returns a reset token, or None.

    A falsy ``auth_token`` binds nothing (and returns None) so callers can
    unconditionally bind without guarding on presence. ``user_id`` rides along
    on the same gate: the token's presence is what marks a request as W-routed,
    and only W-routed traffic may carry W identifiers in its request body. An
    anonymous ``user_id`` binds as no user at all.
    """
    if not auth_token:
        return None
    return _LLM_CREDENTIALS.set(
        LLMCredentials(
            auth_token=auth_token,
            project_id=project_id,
            # The token's own claim wins: it is what the gateway bills by, so
            # preferring it keeps every route of one run on a single end user.
            user_id=_user_id_from_auth_token(auth_token) or _real_user_id(user_id),
        )
    )


def bind_llm_user_id(user_id: Optional[str]) -> Optional[contextvars.Token]:
    """Bind attribution only -- who the work is for, with no JWT to act as.

    The embedding worker runs long after the request that enqueued it and has no
    use for the caller's JWT (embeddings reach the proxy under its own service
    key), but its spend still has to land on the right user. A falsy or anonymous
    ``user_id`` binds nothing, so this is safe to call unconditionally.
    """
    real_user_id = _real_user_id(user_id)
    if real_user_id is None:
        return None
    return _LLM_CREDENTIALS.set(LLMCredentials(user_id=real_user_id))


def reset_llm_credentials(token: Optional[contextvars.Token]) -> None:
    """Reset credentials using a token from :func:`bind_llm_credentials`."""
    if token is not None:
        _LLM_CREDENTIALS.reset(token)


def get_llm_credentials() -> Optional[LLMCredentials]:
    """Return the credentials bound to the current task, if any."""
    return _LLM_CREDENTIALS.get()


def apply_llm_request_metadata(
    kwargs: Dict[str, Any],
    feature_tag: Optional[str] = None,
    tag_without_credentials: bool = False,
) -> None:
    """Merge W request metadata into ``extra_body["metadata"]`` in place.

    Carries who the work is for (``user_id``) and, when given, what it is for
    (a ``feature:`` tag). This is the attribution-only half of
    :func:`apply_llm_credentials`, for calls that reach the LLM proxy under its
    own service key rather than the caller's JWT (embeddings, which the proxy
    does not meter per user but does log). It must not touch auth: overriding
    ``Authorization`` on those routes would replace a valid proxy key with a JWT
    the route does not accept.

    ``feature_tags`` is the key the W gateway itself uses to hand feature tags to
    the proxy's callback, which merges them into the spend log's request tags.
    Traffic that skips the gateway has to write it directly, so this plays the
    part the gateway would have played.

    The two halves are gated differently, because they answer to different
    things. ``user_id`` is a W identifier, so it rides on a bound credential and
    appears only when one exists. A feature tag describes the *call site*, not
    the caller: work with no user behind it -- preset-directory bootstrap, a ROOT
    reindex -- is still worth labelling, and gating the tag on a credential is
    exactly what left those rows untagged. So a caller that knows it is talking
    to the W proxy passes ``tag_without_credentials`` to label unattributed
    traffic too. Callers that cannot know their destination is W leave it False,
    and nothing is sent unless a credential is bound.

    ``metadata`` is a parameter LiteLLM consumes itself, so it never reaches the
    upstream model provider.
    """
    creds = get_llm_credentials()
    if creds is None and not tag_without_credentials:
        return
    user_id = creds.user_id if creds is not None else None
    if not user_id and not feature_tag:
        return

    extra_body = dict(kwargs.get("extra_body") or {})
    metadata = dict(extra_body.get("metadata") or {})
    if user_id:
        metadata[_USER_METADATA_KEY] = str(user_id)
    if feature_tag:
        tags = list(metadata.get(_FEATURE_TAGS_METADATA_KEY) or [])
        if feature_tag not in tags:
            tags.append(feature_tag)
        metadata[_FEATURE_TAGS_METADATA_KEY] = tags
    extra_body["metadata"] = metadata
    kwargs["extra_body"] = extra_body


def apply_llm_credentials(
    kwargs: Dict[str, Any],
    base_headers: Optional[Dict[str, str]] = None,
) -> None:
    """Inject the bound credentials into OpenAI-SDK ``create`` kwargs in place.

    Sets ``extra_headers`` with an ``Authorization: Bearer`` override (which the
    SDK applies on top of the client's static key) and merges ``project_id`` and
    ``metadata.user_id`` into ``extra_body`` (the gateway pops ``project_id``
    from the JSON body before forwarding to LiteLLM). No-op when no credential
    is bound.
    """
    creds = get_llm_credentials()
    if creds is None:
        return

    # An attribution-only binding carries no token; leave the client's own auth
    # in place rather than overriding it with nothing.
    if creds.auth_token:
        headers: Dict[str, str] = dict(base_headers or {})
        existing_headers = kwargs.get("extra_headers")
        if existing_headers:
            headers = {**existing_headers, **headers}
        headers["Authorization"] = f"Bearer {creds.auth_token}"
        kwargs["extra_headers"] = headers

    if creds.project_id:
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body["project_id"] = creds.project_id
        kwargs["extra_body"] = extra_body

    apply_llm_request_metadata(kwargs)
