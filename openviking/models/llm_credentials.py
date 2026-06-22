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

import contextvars
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LLMCredentials:
    """Caller credentials forwarded to the W LLM gateway."""

    auth_token: str
    project_id: Optional[str] = None


_LLM_CREDENTIALS: contextvars.ContextVar[Optional[LLMCredentials]] = contextvars.ContextVar(
    "openviking_llm_credentials",
    default=None,
)


def bind_llm_credentials(
    auth_token: Optional[str],
    project_id: Optional[str] = None,
) -> Optional[contextvars.Token]:
    """Bind credentials for the current task; returns a reset token, or None.

    A falsy ``auth_token`` binds nothing (and returns None) so callers can
    unconditionally bind without guarding on presence.
    """
    if not auth_token:
        return None
    return _LLM_CREDENTIALS.set(LLMCredentials(auth_token=auth_token, project_id=project_id))


def reset_llm_credentials(token: Optional[contextvars.Token]) -> None:
    """Reset credentials using a token from :func:`bind_llm_credentials`."""
    if token is not None:
        _LLM_CREDENTIALS.reset(token)


def get_llm_credentials() -> Optional[LLMCredentials]:
    """Return the credentials bound to the current task, if any."""
    return _LLM_CREDENTIALS.get()


def apply_llm_credentials(
    kwargs: Dict[str, Any],
    base_headers: Optional[Dict[str, str]] = None,
) -> None:
    """Inject the bound credentials into OpenAI-SDK ``create`` kwargs in place.

    Sets ``extra_headers`` with an ``Authorization: Bearer`` override (which the
    SDK applies on top of the client's static key) and merges ``project_id`` into
    ``extra_body`` (the gateway pops it from the JSON body before forwarding to
    LiteLLM). No-op when no credential is bound.
    """
    creds = get_llm_credentials()
    if creds is None:
        return

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
