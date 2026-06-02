# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-project Turbopuffer namespace isolation tests."""

from openviking.storage.vectordb_adapters.turbopuffer_adapter import (
    TurbopufferCollectionAdapter,
)
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


def test_namespace_override_applied():
    a = TurbopufferCollectionAdapter.from_config(_cfg(), namespace_override="context-2-3493")
    assert a._namespace_name == "context-2-3493"


def test_namespace_defaults_to_collection_name():
    a = TurbopufferCollectionAdapter.from_config(_cfg())
    assert a._namespace_name == "context-2"


def test_facade_builds_distinct_namespace_per_account():
    from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend

    be = VikingVectorIndexBackend(_cfg())
    a1 = be._adapter_for_account("3493")
    a2 = be._adapter_for_account("999")
    a_def = be._adapter_for_account("default")
    assert a1._namespace_name == "context-2-3493"
    assert a2._namespace_name == "context-2-999"
    assert a_def is be._shared_adapter
    assert be._adapter_for_account("3493") is a1  # cached
    # All per-project adapters share the shared adapter's single Turbopuffer client.
    assert a1._get_client() is a2._get_client()
    assert a1._get_client() is be._shared_adapter._get_client()
