# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Mock-based unit tests for the Turbopuffer vector backend."""

from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _install_turbopuffer_stub():
    """Ensure a `turbopuffer` module is importable even without the SDK installed."""
    if "turbopuffer" in sys.modules:
        return
    stub = types.ModuleType("turbopuffer")
    stub.Turbopuffer = MagicMock(name="Turbopuffer")
    stub.AsyncTurbopuffer = MagicMock(name="AsyncTurbopuffer")
    sys.modules["turbopuffer"] = stub


_install_turbopuffer_stub()


from openviking.storage.expr import (  # noqa: E402
    And,
    Contains,
    Eq,
    In,
    Or,
    PathScope,
    Range,
)
from openviking.storage.vectordb_adapters import (  # noqa: E402
    TurbopufferCollectionAdapter,
    create_collection_adapter,
)
from openviking.storage.vectordb_adapters.turbopuffer_adapter import (  # noqa: E402
    _map_distance,
)
from openviking_cli.utils.config.vectordb_config import (  # noqa: E402
    TurbopufferConfig,
    VectorDBBackendConfig,
)


def _make_config(**overrides) -> VectorDBBackendConfig:
    tp_overrides = overrides.pop("turbopuffer", None)
    defaults = {"api_key": "test-key", "region": "gcp-us-central1"}
    if tp_overrides:
        defaults.update(tp_overrides)
    return VectorDBBackendConfig(
        backend="turbopuffer",
        name=overrides.pop("name", "test_ns"),
        index_name=overrides.pop("index_name", "default"),
        turbopuffer=TurbopufferConfig(**defaults),
        **overrides,
    )


class TestConfigValidation(unittest.TestCase):
    def test_rejects_missing_api_key(self):
        with patch.dict(os.environ, {"TURBOPUFFER_API_KEY": "env-key"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                VectorDBBackendConfig(
                    backend="turbopuffer",
                    name="ns",
                    turbopuffer=TurbopufferConfig(region="gcp-us-central1"),
                )
            self.assertIn("api_key", str(ctx.exception).lower())

    def test_rejects_missing_region_and_base_url(self):
        with self.assertRaises(ValueError) as ctx:
            VectorDBBackendConfig(
                backend="turbopuffer",
                name="ns",
                turbopuffer=TurbopufferConfig(api_key="k"),
            )
        msg = str(ctx.exception).lower()
        self.assertIn("region", msg)

    def test_accepts_base_url_without_region(self):
        cfg = VectorDBBackendConfig(
            backend="turbopuffer",
            name="ns",
            turbopuffer=TurbopufferConfig(api_key="k", base_url="https://example.tpuf"),
        )
        assert cfg.turbopuffer.base_url == "https://example.tpuf"


class TestFactoryRouting(unittest.TestCase):
    def test_routes_to_turbopuffer_adapter(self):
        cfg = _make_config()
        adapter = create_collection_adapter(cfg)
        self.assertIsInstance(adapter, TurbopufferCollectionAdapter)
        self.assertEqual(adapter.mode, "turbopuffer")
        self.assertEqual(adapter.collection_name, "test_ns")
        self.assertEqual(adapter._namespace_name, "test_ns")
        self.assertEqual(adapter._api_key, "test-key")

    def test_namespace_override_respected(self):
        cfg = _make_config(turbopuffer={"api_key": "k", "region": "r", "namespace": "alt"})
        adapter = create_collection_adapter(cfg)
        self.assertEqual(adapter._namespace_name, "alt")
        self.assertEqual(adapter.collection_name, "test_ns")


class TestDistanceMapping(unittest.TestCase):
    def test_cosine_maps(self):
        self.assertEqual(_map_distance("cosine"), "cosine_distance")
        self.assertEqual(_map_distance("Cosine"), "cosine_distance")
        self.assertEqual(_map_distance("cosine_distance"), "cosine_distance")

    def test_l2_maps(self):
        self.assertEqual(_map_distance("l2"), "euclidean_squared")
        self.assertEqual(_map_distance("euclidean"), "euclidean_squared")

    def test_ip_rejected(self):
        with self.assertRaises(ValueError):
            _map_distance("ip")


class TestFilterCompilation(unittest.TestCase):
    def setUp(self):
        cfg = _make_config()
        self.adapter = create_collection_adapter(cfg)

    def test_eq(self):
        self.assertEqual(
            self.adapter._compile_filter(Eq("status", "active")),
            ("status", "Eq", "active"),
        )

    def test_in(self):
        self.assertEqual(
            self.adapter._compile_filter(In("id", [1, 2, 3])),
            ("id", "In", [1, 2, 3]),
        )

    def test_range_single_bound(self):
        self.assertEqual(
            self.adapter._compile_filter(Range("price", lt=100)),
            ("price", "Lt", 100),
        )

    def test_range_two_bounds(self):
        self.assertEqual(
            self.adapter._compile_filter(Range("price", gte=10, lt=100)),
            ("And", [("price", "Gte", 10), ("price", "Lt", 100)]),
        )

    def test_and_or(self):
        expr = And(
            [
                Eq("category", "tech"),
                Or([Eq("status", "active"), Eq("status", "draft")]),
            ]
        )
        compiled = self.adapter._compile_filter(expr)
        self.assertEqual(
            compiled,
            (
                "And",
                [
                    ("category", "Eq", "tech"),
                    (
                        "Or",
                        [("status", "Eq", "active"), ("status", "Eq", "draft")],
                    ),
                ],
            ),
        )

    def test_contains_to_glob(self):
        self.assertEqual(
            self.adapter._compile_filter(Contains("name", "abc")),
            ("name", "Glob", "*abc*"),
        )

    def test_none_returns_none(self):
        self.assertIsNone(self.adapter._compile_filter(None))

    def test_pathscope_exact_self(self):
        # depth=0 → the node itself, an exact match (not a glob).
        self.assertEqual(
            self.adapter._compile_filter(PathScope("p", "a/b", depth=0)),
            ("p", "Eq", "a/b"),
        )

    def test_pathscope_immediate_children(self):
        # depth=1 → exactly one segment below the prefix.
        self.assertEqual(
            self.adapter._compile_filter(PathScope("p", "a/b", depth=1)),
            ("p", "Glob", "a/b/*"),
        )

    def test_pathscope_bounded_depth(self):
        # depth=2 → exactly two segments below the prefix.
        self.assertEqual(
            self.adapter._compile_filter(PathScope("p", "a/b", depth=2)),
            ("p", "Glob", "a/b/*/*"),
        )

    def test_pathscope_unbounded_subtree(self):
        # depth<0 (the default) → whole subtree; globset ** crosses '/'.
        self.assertEqual(
            self.adapter._compile_filter(PathScope("p", "a/b", depth=-1)),
            ("p", "Glob", "a/b/**"),
        )

    def test_pathscope_uri_field_unbounded_is_recursive(self):
        # uri values get re-encoded by base helpers, but depth=-1 must still
        # produce a recursive ** glob (regression for direct-children-only bug).
        compiled = self.adapter._compile_filter(
            PathScope("uri", "wfs://foo", depth=-1)
        )
        self.assertEqual(compiled[0], "uri")
        self.assertEqual(compiled[1], "Glob")
        self.assertTrue(compiled[2].endswith("/**"))


class TestUpsertAndSchemaApply(unittest.TestCase):
    def setUp(self):
        self.ns_mock = MagicMock()
        self.client_mock = MagicMock()
        self.client_mock.namespace.return_value = self.ns_mock
        # namespaces() returns an iterable of summaries — empty means "does not exist".
        self.client_mock.namespaces.return_value = iter([])
        patcher = patch(
            "openviking.storage.vectordb_adapters.turbopuffer_adapter._build_client",
            return_value=self.client_mock,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        cfg = _make_config()
        self.adapter = create_collection_adapter(cfg)

    def test_create_collection_applies_distance_and_schema_on_first_upsert(self):
        created = self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        self.assertTrue(created)
        # First upsert should include distance_metric + (possibly empty) schema.
        self.adapter.upsert([{"id": "a", "vector": [0.1, 0.2, 0.3], "text": "hello"}])
        self.assertEqual(self.ns_mock.write.call_count, 1)
        write_kwargs = self.ns_mock.write.call_args.kwargs
        self.assertEqual(write_kwargs["distance_metric"], "cosine_distance")
        self.assertEqual(write_kwargs["upsert_rows"][0]["id"], "a")

        # Second upsert should NOT re-send distance_metric (already applied).
        self.adapter.upsert([{"id": "b", "vector": [0.1, 0.2, 0.4]}])
        second_kwargs = self.ns_mock.write.call_args.kwargs
        self.assertNotIn("distance_metric", second_kwargs)
        self.assertNotIn("schema", second_kwargs)

    def test_delete_by_ids(self):
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        # Stub a query response so the adapter's delete-by-filter path is not triggered.
        deleted = self.adapter.delete(ids=["a", "b"])
        self.assertEqual(deleted, 2)
        self.ns_mock.write.assert_called_with(deletes=["a", "b"])

    def test_query_translates_rows(self):
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        row = SimpleNamespace(id="x1", attributes={"text": "hi", "vector": [0.1]}, dist=0.42)
        self.ns_mock.query.return_value = SimpleNamespace(rows=[row], aggregations=None)
        out = self.adapter.query(query_vector=[0.1, 0.2, 0.3], limit=5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "x1")
        self.assertEqual(out[0]["text"], "hi")
        # Vector (ANN) queries return a cosine _distance_ (lower = better); it must be
        # converted to a similarity (higher = better) so the retriever's descending sort
        # ranks nearer records first. cosine_distance 0.42 -> similarity 1 - 0.42 = 0.58.
        self.assertEqual(out[0]["_score"], pytest.approx(0.58))

        # Inspect the call we made to ns.query
        call_kwargs = self.ns_mock.query.call_args.kwargs
        self.assertEqual(call_kwargs["rank_by"], ("vector", "ANN", [0.1, 0.2, 0.3]))
        self.assertEqual(call_kwargs["top_k"], 5)

    def test_vector_query_score_is_similarity_not_distance(self):
        # Regression for inverted ranking: with cosine_distance, the record NEAREST the
        # query (smallest $dist) must receive the HIGHEST _score, so a descending sort
        # surfaces it first — the opposite of returning raw distance.
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        near = SimpleNamespace(id="near", attributes={"text": "near"}, dist=0.20)
        far = SimpleNamespace(id="far", attributes={"text": "far"}, dist=0.80)
        self.ns_mock.query.return_value = SimpleNamespace(rows=[near, far], aggregations=None)
        out = self.adapter.query(query_vector=[0.1, 0.2, 0.3], limit=5)
        by_id = {r["id"]: r["_score"] for r in out}
        self.assertEqual(by_id["near"], pytest.approx(0.80))  # 1 - 0.20
        self.assertEqual(by_id["far"], pytest.approx(0.20))   # 1 - 0.80
        self.assertGreater(by_id["near"], by_id["far"])

    def test_keyword_query_score_left_as_raw_relevance(self):
        # BM25 keyword search returns a relevance score (higher = better), not a vector
        # distance — it must NOT be run through the distance->similarity conversion.
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        row = SimpleNamespace(id="k1", attributes={"text": "hit"}, dist=3.5)
        self.ns_mock.query.return_value = SimpleNamespace(rows=[row], aggregations=None)
        coll = self.adapter._wrap_collection()
        res = coll.search_by_keywords(index_name="default", query="hit", limit=5)
        self.assertEqual(res.data[0].score, pytest.approx(3.5))
        self.assertEqual(
            self.ns_mock.query.call_args.kwargs["rank_by"][1], "BM25"
        )

    def test_count_via_aggregate(self):
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        self.ns_mock.query.return_value = SimpleNamespace(rows=[], aggregations={"_total": 7})
        total = self.adapter.count()
        self.assertEqual(total, 7)
        call_kwargs = self.ns_mock.query.call_args.kwargs
        self.assertEqual(call_kwargs["aggregate_by"], {"_total": ("Count",)})

    def test_query_top_k_clamped_to_tp_max(self):
        # base.delete(filter=...) issues query(limit=100000); top_k must be clamped
        # to TP's 10k ceiling instead of erroring the query → silent empty result.
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        self.ns_mock.query.return_value = SimpleNamespace(rows=[], aggregations=None)
        self.adapter.query(query_vector=[0.1, 0.2, 0.3], limit=100000)
        self.assertEqual(self.ns_mock.query.call_args.kwargs["top_k"], 10000)

    def test_delete_failure_propagates(self):
        # A failed TP delete must raise, not be reported as a successful delete count.
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        self.ns_mock.write.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.adapter.delete(ids=["a", "b"])

    def test_hybrid_query_fuses_dense_and_sparse(self):
        # When both a dense and sparse vector are supplied, two ranked queries run
        # and their results are fused client-side (no silent sparse-only fallback).
        self.adapter.create_collection(
            name="test_ns",
            schema={"Fields": []},
            distance="cosine",
            sparse_weight=0.5,
            index_name="default",
        )

        def _row(rid):
            return SimpleNamespace(id=rid, attributes={"text": rid}, dist=0.1)

        dense = SimpleNamespace(rows=[_row("d1"), _row("d2")], aggregations=None)
        sparse = SimpleNamespace(rows=[_row("d2"), _row("d3")], aggregations=None)
        self.ns_mock.query.side_effect = [dense, sparse]

        out = self.adapter.query(
            query_vector=[0.1, 0.2, 0.3],
            sparse_query_vector={"0": 0.9},
            limit=10,
        )
        self.assertEqual(self.ns_mock.query.call_count, 2)
        rank_bys = [c.kwargs["rank_by"] for c in self.ns_mock.query.call_args_list]
        self.assertEqual(rank_bys[0], ("vector", "ANN", [0.1, 0.2, 0.3]))
        self.assertEqual(rank_bys[1], ("sparse_vector", "SparseKNN", {"0": 0.9}))
        # d2 appears top-ranked in both lists → highest fused score.
        self.assertEqual([r["id"] for r in out], ["d2", "d1", "d3"])


class _PydanticRow:
    """Stand-in for a turbopuffer>=2.x Row: attributes live in ``model_extra``."""

    def __init__(self, id, vector=None, **extra):
        self.id = id
        self.vector = vector
        self.model_extra = dict(extra)


class TestRowExtraction(unittest.TestCase):
    """TP 2.x returns rows whose attributes live in ``model_extra`` (not an
    ``.attributes`` dict). Regression guard: fields/score must be read from it."""

    def test_fields_and_score_from_model_extra(self):
        from openviking.storage.vectordb.collection.turbopuffer_collection import (
            TurbopufferCollection,
        )

        row = _PydanticRow(
            "x1",
            vector=[0.1, 0.2],
            uri="/resources/foo/bar.ts",
            abstract="a tRPC router",
            **{"$dist": 0.42},
        )
        self.assertEqual(TurbopufferCollection._row_id(row), "x1")
        fields = TurbopufferCollection._row_fields(row)
        self.assertEqual(fields.get("uri"), "/resources/foo/bar.ts")
        self.assertEqual(fields.get("abstract"), "a tRPC router")
        self.assertNotIn("$dist", fields)
        self.assertAlmostEqual(TurbopufferCollection._row_score(row), 0.42)

    def test_vectorless_rows_skipped_on_write(self):
        ns = MagicMock()
        client = MagicMock()
        client.namespace.return_value = ns
        from openviking.storage.vectordb.collection.turbopuffer_collection import (
            get_or_create_turbopuffer_collection,
        )

        coll = get_or_create_turbopuffer_collection(
            client=client, namespace_name="ns", distance_metric="cosine_distance"
        )
        ids = coll.upsert_data(
            [
                {"id": "novec"},  # vectorless → skipped from the write
                {"id": "withvec", "vector": [0.1, 0.2, 0.3]},
            ]
        )
        self.assertEqual(ids, ["novec", "withvec"])  # all ids reported handled
        written = ns.write.call_args.kwargs["upsert_rows"]
        self.assertEqual([r["id"] for r in written], ["withvec"])


if __name__ == "__main__":
    unittest.main()
