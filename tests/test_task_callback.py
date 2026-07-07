# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unit tests for the task-completion webhook deliverer."""

import logging

import httpx
import pytest

import openviking.service.task_callback as task_callback_module
from openviking.service.task_callback import deliver_task_callback
from openviking.service.task_tracker import TaskRecord, TaskStatus
from openviking_cli.exceptions import PermissionDeniedError

pytestmark = pytest.mark.asyncio


def _make_task(**overrides) -> TaskRecord:
    defaults = {
        "task_id": "task-1",
        "task_type": "resource_ingest",
        "status": TaskStatus.COMPLETED,
        "resource_id": "res-1",
        "callback_url": "https://caller.example.com/hook",
        "callback_token": "tok-secret",
        "result": {"root_uri": "wfs://resources/res-1", "internal_debug": "should-not-leak"},
        "error": None,
    }
    defaults.update(overrides)
    return TaskRecord(**defaults)


@pytest.fixture(autouse=True)
def _bypass_network_guard(monkeypatch):
    """Most tests here focus on delivery/retry behavior, not SSRF policy
    (that's covered by tests/misc/test_network_guard.py). Dedicated tests
    below re-enable/override the guard to verify the integration point."""
    monkeypatch.setattr(task_callback_module, "ensure_public_remote_target", lambda url: None)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Never actually sleep 1s/3s in tests."""
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(task_callback_module.asyncio, "sleep", fake_sleep)
    return sleeps


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _install_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)

    def factory(**kwargs):
        kwargs.pop("timeout", None)
        return _REAL_ASYNC_CLIENT(transport=transport, timeout=10, **kwargs)

    monkeypatch.setattr(task_callback_module.httpx, "AsyncClient", factory)


_LOGGER_NAME = "openviking.service.task_callback"


@pytest.fixture
def callback_caplog(caplog):
    """caplog that actually captures task_callback's logger.

    ``get_logger`` configures module loggers with their own handler and
    ``propagate = False`` (see openviking_cli/utils/logger.py), so records
    never reach caplog's root-attached handler via ``at_level`` alone. Attach
    caplog's handler directly to the module logger for the duration of the test.
    """
    target = logging.getLogger(_LOGGER_NAME)
    original_propagate = target.propagate
    original_level = target.level
    target.addHandler(caplog.handler)
    target.propagate = True
    target.setLevel(logging.WARNING)
    try:
        yield caplog
    finally:
        target.removeHandler(caplog.handler)
        target.propagate = original_propagate
        target.setLevel(original_level)


# ── Success / retry behavior ──


async def test_delivers_on_first_2xx_response(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task())

    assert len(calls) == 1


async def test_retries_on_failure_then_succeeds(monkeypatch, _no_real_sleep):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(500)
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task())

    assert len(calls) == 3
    assert _no_real_sleep == [1, 3]


async def test_all_failures_logs_warning_without_raising(monkeypatch, callback_caplog):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task())  # must not raise

    assert len(calls) == 3
    warning_messages = [
        r.getMessage() for r in callback_caplog.records if r.levelno == logging.WARNING
    ]
    assert any("task-1" in m for m in warning_messages)


async def test_never_logs_the_callback_token(monkeypatch, callback_caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task(callback_token="super-secret-token"))

    assert len(callback_caplog.records) > 0  # sanity: something was actually captured
    for record in callback_caplog.records:
        assert "super-secret-token" not in record.getMessage()


# ── Request contents ──


async def test_includes_token_header_when_set(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task(callback_token="tok-secret"))

    assert captured["headers"]["X-OpenViking-Callback-Token"] == "tok-secret"
    assert captured["headers"]["Content-Type"] == "application/json"


async def test_omits_token_header_when_not_set(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task(callback_token=None))

    assert "X-OpenViking-Callback-Token" not in captured["headers"]


async def test_payload_trims_result_to_root_uri(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    import json

    await deliver_task_callback(
        _make_task(result={"root_uri": "wfs://resources/res-1", "secret_blob": "x" * 1000})
    )

    body = json.loads(captured["body"])
    assert body["task_id"] == "task-1"
    assert body["task_type"] == "resource_ingest"
    assert body["status"] == "completed"
    assert body["resource_id"] == "res-1"
    assert body["error"] is None
    assert body["result"] == {"root_uri": "wfs://resources/res-1"}
    assert "secret_blob" not in json.dumps(body)


async def test_payload_omits_result_when_no_root_uri(monkeypatch):
    import json

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task(result={"memories_extracted": 3}))

    body = json.loads(captured["body"])
    assert body["result"] is None


async def test_payload_status_is_failed_string_for_failed_task(monkeypatch):
    import json

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(
        _make_task(status=TaskStatus.FAILED, error="LLM timeout", result=None)
    )

    body = json.loads(captured["body"])
    assert body["status"] == "failed"
    assert body["error"] == "LLM timeout"


# ── No-ops ──


async def test_noop_when_no_callback_url(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not send a request when callback_url is unset")

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task(callback_url=None))  # must not raise


# ── SSRF guard integration ──


async def test_skips_delivery_when_guard_rejects_url(monkeypatch, callback_caplog):
    def raise_guard(url):
        raise PermissionDeniedError("loopback destination rejected")

    monkeypatch.setattr(task_callback_module, "ensure_public_remote_target", raise_guard)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not send a request when guard rejects the URL")

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task())  # must not raise

    assert any("task-1" in r.getMessage() for r in callback_caplog.records)


async def test_guard_receives_the_callback_url(monkeypatch):
    seen = []
    monkeypatch.setattr(task_callback_module, "ensure_public_remote_target", seen.append)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    _install_transport(monkeypatch, handler)

    await deliver_task_callback(_make_task(callback_url="https://caller.example.com/hook"))

    assert seen == ["https://caller.example.com/hook"]
