# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ResourceService.add_resource forwards callback kwargs to task creation."""

from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.resource_service import ResourceService
from openviking_cli.session.user_id import UserIdentifier


class MockResourceProcessor:
    async def process_resource(self, **kwargs):
        return {"root_uri": kwargs.get("to", "wfs://resources/test")}


class MockSkillProcessor:
    async def process_skill(self, **kwargs):
        return {"status": "ok"}


class MockVikingFS:
    pass


class MockVikingDB:
    pass


class SpyTaskTracker:
    def __init__(self):
        self.create_calls: list[tuple[tuple, dict]] = []

    async def create(self, *args, **kwargs):
        self.create_calls.append((args, kwargs))
        return SimpleNamespace(task_id="task-1")

    async def start(self, *args, **kwargs):
        pass

    async def complete(self, *args, **kwargs):
        pass

    async def fail(self, *args, **kwargs):
        pass


@pytest.fixture
def resource_service() -> ResourceService:
    return ResourceService(
        vikingdb=MockVikingDB(),
        viking_fs=MockVikingFS(),
        resource_processor=MockResourceProcessor(),
        skill_processor=MockSkillProcessor(),
    )


@pytest.fixture
def request_context() -> RequestContext:
    return RequestContext(
        user=UserIdentifier("test_account", "test_user", "test_agent"),
        role=Role.USER,
    )


@pytest.fixture
def spy_tracker(monkeypatch) -> SpyTaskTracker:
    tracker = SpyTaskTracker()
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker", lambda: tracker
    )
    return tracker


@pytest.mark.asyncio
async def test_add_resource_forwards_callback_kwargs_to_task_creation(
    resource_service: ResourceService,
    request_context: RequestContext,
    spy_tracker: SpyTaskTracker,
):
    await resource_service.add_resource(
        path="/test/path",
        ctx=request_context,
        callback_url="http://localhost:8080/callback",
        callback_token="CBTOKEN",
    )

    assert spy_tracker.create_calls, "task_tracker.create was never called"
    _, kwargs = spy_tracker.create_calls[0]
    assert kwargs["callback_url"] == "http://localhost:8080/callback"
    assert kwargs["callback_token"] == "CBTOKEN"


@pytest.mark.asyncio
async def test_add_resource_defaults_callback_kwargs_to_none(
    resource_service: ResourceService,
    request_context: RequestContext,
    spy_tracker: SpyTaskTracker,
):
    await resource_service.add_resource(path="/test/path", ctx=request_context)

    assert spy_tracker.create_calls, "task_tracker.create was never called"
    _, kwargs = spy_tracker.create_calls[0]
    assert kwargs.get("callback_url") is None
    assert kwargs.get("callback_token") is None
