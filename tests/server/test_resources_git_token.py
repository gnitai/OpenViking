# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Integration: the X-OpenViking-Git-Token header reaches the service layer.

Drives the *real* resources router through ASGI so the new ``Header`` route
parameter and the full request wiring are exercised (a malformed route signature
would fail at router registration / request handling, not at import).
"""

import httpx
import pytest
from fastapi import FastAPI

from openviking.server.dependencies import ensure_project_ready, set_service
from openviking.server.identity import RequestContext, Role
from openviking.server.routers.resources import router as resources_router
from openviking_cli.session.user_id import UserIdentifier


class _Resources:
    def __init__(self):
        self.calls: list[dict] = []

    async def add_resource(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "success", "root_uri": "wfs://resources/r"}


class _Service:
    def __init__(self):
        self.resources = _Resources()


def _app(service: _Service) -> FastAPI:
    app = FastAPI()
    # include_router analyzes the endpoint signature (incl. Header) -- a bad
    # signature raises here, validating route registration.
    app.include_router(resources_router)
    set_service(service)
    ctx = RequestContext(user=UserIdentifier("acct", "user", "agent"), role=Role.ADMIN)
    app.dependency_overrides[ensure_project_ready] = lambda: ctx
    return app


async def _post(app: FastAPI, headers: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        return await c.post(
            "/api/v1/resources",
            json={"path": "https://github.com/octocat/Hello-World"},
            headers=headers,
        )


@pytest.mark.asyncio
async def test_git_token_header_reaches_service():
    svc = _Service()
    app = _app(svc)
    resp = await _post(app, {"X-OpenViking-Git-Token": "HEADERTOKEN"})
    assert resp.status_code == 200, resp.text
    assert svc.resources.calls, "service.add_resource was never called"
    assert svc.resources.calls[0]["git_auth_token"] == "HEADERTOKEN"


@pytest.mark.asyncio
async def test_no_header_means_none_token():
    svc = _Service()
    app = _app(svc)
    resp = await _post(app, {})
    assert resp.status_code == 200, resp.text
    assert svc.resources.calls[0]["git_auth_token"] is None
