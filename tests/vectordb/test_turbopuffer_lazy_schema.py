# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lazy per-project schema application via ``ensure_collection``.

Task 3: a project gets its OWN Turbopuffer namespace
(``{collection}-{project_id}``). That namespace has no schema until
``create_collection`` runs, and Turbopuffer strips every attribute (including
``vector``) from writes while ``Fields`` is empty. So ``ensure_collection``
must apply the FULL schema to the PER-PROJECT backend (not the default one),
every process.

Note on test fidelity: the existing MagicMock-based Turbopuffer doubles do not
model Fields-driven attribute stripping (that lives in the adapter wrap layer
and in ``_SingleAccountBackend._filter_known_fields`` reading
``get_meta_data()["Fields"]``). So we verify the sanctioned invariant directly:
``ensure_collection`` invokes ``create_collection`` on the per-account backend
(distinct from the default backend) with a schema whose ``Fields`` is non-empty
and includes the ``vector`` field carrying the configured dimension.
"""

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.vectordb_config import (
    TurbopufferConfig,
    VectorDBBackendConfig,
)


def _cfg():
    return VectorDBBackendConfig(
        backend="turbopuffer",
        name="context-2",
        dimension=1024,
        turbopuffer=TurbopufferConfig(api_key="k", region="aws-us-east-1"),
    )


def _ctx(account: str, user: str) -> RequestContext:
    return RequestContext(
        user=UserIdentifier(account, user, "default"),
        role=Role.ROOT,
    )


@pytest.mark.asyncio
async def test_ensure_collection_applies_schema_to_per_project_backend(monkeypatch):
    be = VikingVectorIndexBackend(_cfg())

    captured = {}

    project_backend = be._get_backend_for_context(_ctx("3493", "m1"))
    default_backend = be._get_default_backend()
    assert project_backend is not default_backend  # distinct per-project namespace

    async def _fake_create(name, schema):
        captured["self"] = "called"
        captured["name"] = name
        captured["schema"] = schema
        return True

    monkeypatch.setattr(project_backend, "create_collection", _fake_create)

    # The default backend's create_collection MUST NOT be touched.
    async def _fail_default(name, schema):  # pragma: no cover
        raise AssertionError("default backend create_collection should not be called")

    monkeypatch.setattr(default_backend, "create_collection", _fail_default)

    result = await be.ensure_collection(ctx=_ctx("3493", "m1"))

    assert result is True
    assert captured["name"] == "context-2"
    fields = captured["schema"]["Fields"]
    assert isinstance(fields, list) and len(fields) > 0
    by_name = {f["FieldName"]: f for f in fields}
    assert "vector" in by_name
    assert by_name["vector"]["Dim"] == 1024
    assert "account_id" in by_name


@pytest.mark.asyncio
async def test_ensure_collection_returns_backend_result(monkeypatch):
    be = VikingVectorIndexBackend(_cfg())
    project_backend = be._get_backend_for_context(_ctx("999", "m1"))

    async def _fake_create(name, schema):
        return False

    monkeypatch.setattr(project_backend, "create_collection", _fake_create)

    assert await be.ensure_collection(ctx=_ctx("999", "m1")) is False
