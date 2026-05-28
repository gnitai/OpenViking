# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Live integration tests for the Turbopuffer vector backend.

Skipped unless TURBOPUFFER_API_KEY is set in the environment. Set
TURBOPUFFER_REGION (default: gcp-us-central1) to pick a region.
"""

from __future__ import annotations

import os
import time
import unittest
import uuid

from openviking.storage.vectordb_adapters import create_collection_adapter
from openviking_cli.utils.config.vectordb_config import (
    TurbopufferConfig,
    VectorDBBackendConfig,
)


def _live_creds() -> bool:
    return bool(os.environ.get("TURBOPUFFER_API_KEY"))


@unittest.skipUnless(_live_creds(), "TURBOPUFFER_API_KEY not set; skipping live test")
class TestTurbopufferLive(unittest.TestCase):
    """Full CRUD cycle against a real Turbopuffer namespace."""

    def setUp(self):
        self.namespace = f"openviking-it-{uuid.uuid4().hex[:8]}"
        self.config = VectorDBBackendConfig(
            backend="turbopuffer",
            name=self.namespace,
            index_name="default",
            distance_metric="cosine",
            turbopuffer=TurbopufferConfig(
                api_key=os.environ["TURBOPUFFER_API_KEY"],
                region=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
                bm25_field="text",
            ),
        )
        self.adapter = create_collection_adapter(self.config)
        self.adapter.create_collection(
            name=self.namespace,
            schema={
                "Fields": [
                    {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
                    {"FieldName": "vector", "FieldType": "vector", "Dim": 3},
                    {"FieldName": "text", "FieldType": "string"},
                    {"FieldName": "category", "FieldType": "string"},
                    {"FieldName": "price", "FieldType": "float"},
                ],
                "ScalarIndex": ["category", "price"],
            },
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )

    def tearDown(self):
        try:
            self.adapter.drop_collection()
        except Exception:
            pass

    def test_full_cycle(self):
        rows = [
            {
                "id": "a",
                "vector": [1.0, 0.0, 0.0],
                "text": "laptop pro",
                "category": "electronics",
                "price": 999.0,
            },
            {
                "id": "b",
                "vector": [0.0, 1.0, 0.0],
                "text": "desk chair",
                "category": "furniture",
                "price": 249.0,
            },
            {
                "id": "c",
                "vector": [0.0, 0.0, 1.0],
                "text": "usb hub",
                "category": "electronics",
                "price": 39.0,
            },
        ]
        ids = self.adapter.upsert(rows)
        self.assertEqual(set(ids), {"a", "b", "c"})

        # Vector query
        hits = self.adapter.query(query_vector=[1.0, 0.0, 0.0], limit=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "a")

        # Filter — wait a tick for indexing in case of write/read lag
        time.sleep(1)
        from openviking.storage.expr import Eq

        electronics = self.adapter.query(
            query_vector=[1.0, 0.0, 0.0],
            filter=Eq("category", "electronics"),
            limit=5,
        )
        electronic_ids = {row["id"] for row in electronics}
        self.assertTrue(electronic_ids.issubset({"a", "c"}))

        # Count
        total = self.adapter.count()
        self.assertEqual(total, 3)

        # Fetch
        fetched = self.adapter.get(["a", "b"])
        self.assertEqual({row["id"] for row in fetched}, {"a", "b"})

        # Delete by id
        deleted = self.adapter.delete(ids=["c"])
        self.assertEqual(deleted, 1)


if __name__ == "__main__":
    unittest.main()
