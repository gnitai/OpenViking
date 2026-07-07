# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Outbound webhook delivery for terminal task-tracker transitions.

When a background task completes or fails, the caller that registered a
``callback_url`` on the task is notified via an HTTP POST. Delivery is
best-effort: it retries a handful of times and then gives up, and it must
never raise or otherwise interfere with task completion.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

from openviking.utils.network_guard import ensure_public_remote_target
from openviking_cli.exceptions import PermissionDeniedError
from openviking_cli.utils.logger import get_logger

if TYPE_CHECKING:
    from openviking.service.task_tracker import TaskRecord

logger = get_logger(__name__)

_REQUEST_TIMEOUT_SECONDS = 10
# Attempt, sleep 1s, attempt, sleep 3s, attempt, give up.
_RETRY_SLEEPS = (1, 3)
_CALLBACK_TOKEN_HEADER = "X-OpenViking-Callback-Token"


def _is_callback_enabled() -> bool:
    """Check the server-config kill switch. Defaults to enabled if unset."""
    try:
        from openviking.server.dependencies import get_server_config

        config = get_server_config()
    except Exception:
        return True
    if config is None:
        return True
    return getattr(config, "task_callback_enabled", True)


def _build_payload(task: "TaskRecord") -> Dict[str, Any]:
    status = task.status
    result = task.result
    trimmed_result: Optional[Dict[str, Any]] = None
    if isinstance(result, dict) and "root_uri" in result:
        trimmed_result = {"root_uri": result["root_uri"]}
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": status.value if hasattr(status, "value") else status,
        "resource_id": task.resource_id,
        "error": task.error,
        "result": trimmed_result,
    }


async def deliver_task_callback(task: "TaskRecord") -> None:
    """POST a completion/failure notification to ``task.callback_url``.

    Best-effort delivery: up to 3 attempts (sleeping 1s then 3s between
    attempts), any 2xx response counts as delivered. Never raises — failures
    are logged as a warning without ever including the callback token.
    """
    if not task.callback_url:
        return

    if not _is_callback_enabled():
        logger.debug(
            "[task_callback] Callback delivery disabled via config; skipping task %s",
            task.task_id,
        )
        return

    try:
        ensure_public_remote_target(task.callback_url)
    except PermissionDeniedError as exc:
        logger.warning(
            "[task_callback] Skipping callback for task %s: URL rejected by network guard (%s)",
            task.task_id,
            exc,
        )
        return
    except Exception:
        logger.warning(
            "[task_callback] Skipping callback for task %s: callback_url failed validation",
            task.task_id,
        )
        return

    payload = _build_payload(task)
    headers = {"Content-Type": "application/json"}
    if task.callback_token:
        headers[_CALLBACK_TOKEN_HEADER] = task.callback_token

    last_error = "unknown error"
    attempts = len(_RETRY_SLEEPS) + 1
    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(task.callback_url, json=payload, headers=headers)
            if 200 <= response.status_code < 300:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # network errors, timeouts, etc.
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < len(_RETRY_SLEEPS):
            await asyncio.sleep(_RETRY_SLEEPS[attempt])

    logger.warning(
        "[task_callback] Failed to deliver callback for task %s to %s after %d attempts: %s",
        task.task_id,
        task.callback_url,
        attempts,
        last_error,
    )
