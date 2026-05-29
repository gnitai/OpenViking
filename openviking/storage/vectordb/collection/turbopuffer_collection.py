# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Turbopuffer ICollection implementation.

Wraps a `turbopuffer.Turbopuffer().namespace(...)` handle and translates
ICollection calls to Turbopuffer SDK calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openviking.storage.vectordb.collection.collection import Collection, ICollection
from openviking.storage.vectordb.collection.result import (
    AggregateResult,
    DataItem,
    FetchDataInCollectionResult,
    SearchItemResult,
    SearchResult,
)
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

# Turbopuffer caps `top_k` / `limit.total` at 10,000; larger values error the query.
_TP_MAX_TOP_K = 10000

# Default hybrid fusion weight applied to the sparse ranking when the namespace was
# loaded (rather than created) this session, so `sparse_weight` wasn't recorded.
_DEFAULT_SPARSE_WEIGHT = 0.5

# Rank constant for reciprocal-rank fusion. Larger => flatter weighting of top ranks.
_RRF_K = 60


def _import_turbopuffer():
    """Lazy import so the SDK is only required at runtime."""
    try:
        import turbopuffer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Turbopuffer backend requires the 'turbopuffer' package. "
            "Install it with `pip install turbopuffer`."
        ) from e
    return turbopuffer


def _build_client(
    *,
    api_key: Optional[str],
    region: Optional[str],
    base_url: Optional[str],
):
    """Construct a `turbopuffer.Turbopuffer` client with the configured creds."""
    tpuf_mod = _import_turbopuffer()
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if region:
        kwargs["region"] = region
    if base_url:
        kwargs["base_url"] = base_url
    return tpuf_mod.Turbopuffer(**kwargs)


def get_or_create_turbopuffer_collection(
    *,
    client: Any,
    namespace_name: str,
    distance_metric: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    bm25_field: str = "text",
    meta_data: Optional[Dict[str, Any]] = None,
) -> Collection:
    """Return a wrapped `Collection` bound to the given Turbopuffer namespace."""
    return Collection(
        TurbopufferCollection(
            client=client,
            namespace_name=namespace_name,
            distance_metric=distance_metric,
            schema=schema,
            bm25_field=bm25_field,
            meta_data=meta_data,
        )
    )


class TurbopufferCollection(ICollection):
    """ICollection implementation backed by a Turbopuffer namespace.

    Turbopuffer namespaces materialise on the first write call. Schema and
    distance metric are stashed in the instance and applied with the next
    `upsert_data` that has rows. Until then they may be reassigned by
    `create_index`.
    """

    def __init__(
        self,
        *,
        client: Any,
        namespace_name: str,
        distance_metric: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        bm25_field: str = "text",
        meta_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._client = client
        self._namespace_name = namespace_name
        self._ns = client.namespace(namespace_name)
        self._distance_metric = distance_metric
        self._schema: Dict[str, Any] = dict(schema or {})
        self._bm25_field = bm25_field
        self._meta_data: Dict[str, Any] = dict(meta_data or {})
        self._meta_data.setdefault("CollectionName", namespace_name)
        self._schema_applied = False
        self._index_name: Optional[str] = None
        self._index_meta: Dict[str, Any] = {}
        self._synthesized_fields: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------ meta

    def _known_fields(self) -> List[Dict[str, Any]]:
        """Field definitions used by callers to gate writes to known columns.

        On the create path these come from the OV schema passed in ``meta_data``.
        On the load path (namespace already exists, no schema handed in) we derive
        the field names from the remote Turbopuffer schema so writes are not
        stripped of their attributes. Derived live each call rather than cached:
        the remote schema grows as attributes are first written, and caching an
        early incomplete view would permanently strip later fields.
        """
        existing = self._meta_data.get("Fields")
        if existing:
            return existing
        try:
            schema = self._ns.schema()
            keys = list(schema.keys()) if hasattr(schema, "keys") else []
            return [{"FieldName": k} for k in keys]
        except Exception:
            # Namespace may not exist yet.
            return []

    def update(
        self,
        fields: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        if fields:
            self._meta_data.update(fields)
        if description is not None:
            self._meta_data["Description"] = description

    def get_meta_data(self) -> Dict[str, Any]:
        merged = dict(self._meta_data)
        if not merged.get("Fields"):
            fields = self._known_fields()
            if fields:
                merged["Fields"] = fields
        try:
            remote = self._ns.metadata()
        except Exception:
            return merged
        if isinstance(remote, dict):
            merged.update(remote)
        else:
            for attr in ("schema", "distance_metric", "approx_count"):
                value = getattr(remote, attr, None)
                if value is not None:
                    merged[attr] = value
        return merged

    def close(self):
        pass

    def drop(self):
        try:
            self._ns.delete_all()
        except Exception:
            logger.exception("Failed to drop turbopuffer namespace %s", self._namespace_name)
            raise
        self._schema_applied = False

    # ----------------------------------------------------------------- index

    def create_index(self, index_name: str, meta_data: Dict[str, Any]):
        self._index_name = index_name
        self._index_meta = dict(meta_data)
        distance = meta_data.get("distance_metric")
        if distance:
            self._distance_metric = distance
        schema = meta_data.get("schema")
        if schema:
            self._schema = dict(schema)
        return None

    def has_index(self, index_name: str) -> bool:
        return self._index_name == index_name

    def get_index(self, index_name: str) -> Optional[Dict[str, Any]]:
        if self.has_index(index_name):
            return dict(self._index_meta)
        return None

    def update_index(
        self,
        index_name: str,
        scalar_index: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        if not self.has_index(index_name):
            return
        if scalar_index is not None:
            self._index_meta["scalar_index_fields"] = scalar_index
        if description is not None:
            self._index_meta["Description"] = description

    def get_index_meta_data(self, index_name: str) -> Dict[str, Any]:
        if self.has_index(index_name):
            return dict(self._index_meta)
        return {}

    def list_indexes(self) -> List[str]:
        return [self._index_name] if self._index_name else []

    def drop_index(self, index_name: str):
        if self.has_index(index_name):
            self._index_name = None
            self._index_meta = {}

    # ------------------------------------------------------------------ data

    def upsert_data(self, data_list: List[Dict[str, Any]], ttl: int = 0):
        if ttl:
            logger.warning("Turbopuffer does not support per-row TTL; ignoring ttl=%s", ttl)
        if not data_list:
            return []
        # Turbopuffer namespaces with a vector/ANN attribute require every row to
        # carry a vector. OpenViking writes some vectorless navigational rows (e.g.
        # empty preset scope roots whose summaries have no embeddable text). Those
        # cannot live in a vector store, so drop them from the write. Their ids are
        # still reported as handled (the pipeline expects every id back), so a later
        # fetch_data/get of such an id returns nothing — the row was never stored.
        rows = [r for r in data_list if r.get("vector") or r.get("sparse_vector")]
        skipped = len(data_list) - len(rows)
        if skipped:
            logger.debug(
                "Dropped %d vectorless row(s) from the write to turbopuffer namespace "
                "%s (reported as handled but not stored)",
                skipped,
                self._namespace_name,
            )
        if not rows:
            return [row.get("id") for row in data_list]
        kwargs: Dict[str, Any] = {"upsert_rows": rows}
        if not self._schema_applied:
            if self._distance_metric:
                kwargs["distance_metric"] = self._distance_metric
            if self._schema:
                kwargs["schema"] = self._schema
        self._ns.write(**kwargs)
        self._schema_applied = True
        return [row.get("id") for row in data_list]

    def fetch_data(self, primary_keys: List[Any]) -> FetchDataInCollectionResult:
        result = FetchDataInCollectionResult()
        if not primary_keys:
            return result
        keys = list(primary_keys)
        try:
            response = self._ns.query(
                filters=("id", "In", keys),
                top_k=min(_TP_MAX_TOP_K, max(1, len(keys))),
                rank_by=("id", "asc"),
                include_attributes=True,
            )
        except Exception:
            logger.exception("Turbopuffer fetch_data failed")
            result.ids_not_exist = keys
            return result
        seen: set = set()
        rows = self._rows(response)
        for row in rows:
            row_id = self._row_id(row)
            seen.add(row_id)
            result.items.append(DataItem(id=row_id, fields=self._row_fields(row)))
        result.ids_not_exist = [pk for pk in keys if pk not in seen]
        return result

    def delete_data(self, primary_keys: List[Any]):
        if not primary_keys:
            return
        try:
            self._ns.write(deletes=list(primary_keys))
        except Exception:
            logger.exception("Turbopuffer delete_data failed")
            raise

    def delete_all_data(self):
        try:
            self._ns.delete_all()
        except Exception:
            logger.exception(
                "Failed to delete_all() turbopuffer namespace %s",
                self._namespace_name,
            )
            raise
        self._schema_applied = False

    # ---------------------------------------------------------------- search

    def search_by_vector(
        self,
        index_name: str,
        dense_vector: Optional[List[float]] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[Dict[str, float]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        has_dense = bool(dense_vector)
        has_sparse = bool(sparse_vector)
        if not has_dense and not has_sparse:
            return SearchResult()
        if has_dense and has_sparse:
            # Turbopuffer ranks by a single function per query, so true hybrid search
            # means running the dense and sparse rankings separately and fusing them
            # client-side (see TP docs: "for hybrid search, multi-queries can be
            # used and combined client-side").
            return self._run_hybrid_query(
                dense_vector=list(dense_vector or []),
                sparse_vector=sparse_vector or {},
                limit=limit,
                offset=offset,
                filters=filters,
                include_attributes=output_fields,
            )
        if has_sparse:
            sparse = {str(k): float(v) for k, v in (sparse_vector or {}).items()}
            rank_by: Any = ("sparse_vector", "SparseKNN", sparse)
        else:
            rank_by = ("vector", "ANN", list(dense_vector or []))
        return self._run_query(
            rank_by=rank_by,
            limit=limit,
            offset=offset,
            filters=filters,
            include_attributes=output_fields,
        )

    def search_by_keywords(
        self,
        index_name: str,
        keywords: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        text = query
        if not text and keywords:
            text = " ".join(str(k) for k in keywords)
        if not text:
            return SearchResult()
        rank_by = (self._bm25_field, "BM25", text)
        return self._run_query(
            rank_by=rank_by,
            limit=limit,
            offset=offset,
            filters=filters,
            include_attributes=output_fields,
        )

    def search_by_id(
        self,
        index_name: str,
        id: Any,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        try:
            seed = self._ns.query(
                filters=("id", "Eq", id),
                top_k=1,
                rank_by=("id", "asc"),
                include_attributes=["vector"],
            )
        except Exception:
            logger.exception("Turbopuffer search_by_id seed lookup failed")
            return SearchResult()
        rows = self._rows(seed)
        if not rows:
            return SearchResult()
        vec = self._row_vector(rows[0])
        if vec is None:
            return SearchResult()
        return self.search_by_vector(
            index_name=index_name,
            dense_vector=list(vec),
            limit=limit,
            offset=offset,
            filters=filters,
            output_fields=output_fields,
        )

    def search_by_multimodal(
        self,
        index_name: str,
        text: Optional[str],
        image: Optional[Any],
        video: Optional[Any],
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        raise NotImplementedError("Turbopuffer does not support multimodal vectorization")

    def search_by_random(
        self,
        index_name: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        # Turbopuffer has no native random sampling; use id-ordered as a
        # deterministic fallback that still respects filters.
        return self._run_query(
            rank_by=("id", "asc"),
            limit=limit,
            offset=offset,
            filters=filters,
            include_attributes=output_fields,
        )

    def search_by_scalar(
        self,
        index_name: str,
        field: str,
        order: Optional[str] = "desc",
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        order_str = "desc" if (order or "desc").lower() == "desc" else "asc"
        return self._run_query(
            rank_by=(field, order_str),
            limit=limit,
            offset=offset,
            filters=filters,
            include_attributes=output_fields,
        )

    def aggregate_data(
        self,
        index_name: str,
        op: str = "count",
        field: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        cond: Optional[Dict[str, Any]] = None,
    ) -> AggregateResult:
        if op != "count":
            logger.warning(
                "Turbopuffer aggregate only supports 'count'; got %r — returning empty",
                op,
            )
            return AggregateResult(agg={}, op=op, field=field)
        try:
            response = self._ns.query(
                filters=filters,
                aggregate_by={"_total": ("Count",)},
            )
        except Exception:
            logger.exception("Turbopuffer aggregate_data failed")
            return AggregateResult(agg={}, op=op, field=field)
        agg = getattr(response, "aggregations", None) or {}
        if not isinstance(agg, dict):
            agg = dict(agg) if hasattr(agg, "items") else {}
        return AggregateResult(agg=dict(agg), op=op, field=field)

    # ----------------------------------------------------------------- inner

    def _query_rows(
        self,
        *,
        rank_by: Any,
        top_k: int,
        filters: Optional[Any],
        include_attributes: Optional[List[str]],
    ) -> List[Any]:
        """Issue a single ranked query and return its rows (empty on failure)."""
        kwargs: Dict[str, Any] = {"rank_by": rank_by, "top_k": top_k}
        if filters:
            kwargs["filters"] = filters
        kwargs["include_attributes"] = (
            list(include_attributes) if include_attributes else True
        )
        try:
            response = self._ns.query(**kwargs)
        except Exception:
            logger.exception("Turbopuffer query failed")
            return []
        return self._rows(response)

    def _run_query(
        self,
        *,
        rank_by: Any,
        limit: int,
        offset: int,
        filters: Optional[Any],
        include_attributes: Optional[List[str]],
    ) -> SearchResult:
        top_k = min(_TP_MAX_TOP_K, max(1, limit + max(0, offset)))
        rows = self._query_rows(
            rank_by=rank_by,
            top_k=top_k,
            filters=filters,
            include_attributes=include_attributes,
        )
        if offset:
            rows = rows[offset:]
        rows = rows[:limit]
        items = [
            SearchItemResult(
                id=self._row_id(row),
                fields=self._row_fields(row),
                score=self._row_score(row),
            )
            for row in rows
        ]
        return SearchResult(data=items)

    def _run_hybrid_query(
        self,
        *,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        limit: int,
        offset: int,
        filters: Optional[Any],
        include_attributes: Optional[List[str]],
    ) -> SearchResult:
        """Run dense + sparse rankings separately and fuse them client-side."""
        sparse = {str(k): float(v) for k, v in sparse_vector.items()}
        top_k = min(_TP_MAX_TOP_K, max(1, limit + max(0, offset)))
        dense_rows = self._query_rows(
            rank_by=("vector", "ANN", dense_vector),
            top_k=top_k,
            filters=filters,
            include_attributes=include_attributes,
        )
        sparse_rows = self._query_rows(
            rank_by=("sparse_vector", "SparseKNN", sparse),
            top_k=top_k,
            filters=filters,
            include_attributes=include_attributes,
        )
        try:
            weight = float(self._index_meta.get("sparse_weight", _DEFAULT_SPARSE_WEIGHT))
        except (TypeError, ValueError):
            weight = _DEFAULT_SPARSE_WEIGHT
        fused = self._fuse_rrf(dense_rows, sparse_rows, sparse_weight=weight)
        if offset:
            fused = fused[offset:]
        fused = fused[:limit]
        items = [
            SearchItemResult(id=rid, fields=fields, score=score)
            for rid, fields, score in fused
        ]
        return SearchResult(data=items)

    @classmethod
    def _fuse_rrf(
        cls,
        dense_rows: List[Any],
        sparse_rows: List[Any],
        *,
        sparse_weight: float,
    ) -> List[Any]:
        """Reciprocal-rank-fuse two rank-ordered row lists into (id, fields, score).

        RRF is scale-free, so it sidesteps the distance-vs-score scale mismatch
        between ANN and SparseKNN. ``sparse_weight`` splits the contribution
        between the two rankings (0 => dense only, 1 => sparse only).
        """
        w_sparse = min(1.0, max(0.0, sparse_weight))
        w_dense = 1.0 - w_sparse
        scores: Dict[Any, float] = {}
        fields: Dict[Any, Dict[str, Any]] = {}
        for weight, rows in ((w_dense, dense_rows), (w_sparse, sparse_rows)):
            for rank, row in enumerate(rows):
                rid = cls._row_id(row)
                if rid is None:
                    continue
                scores[rid] = scores.get(rid, 0.0) + weight / (_RRF_K + rank + 1)
                if rid not in fields:
                    fields[rid] = cls._row_fields(row)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(rid, fields.get(rid, {}), score) for rid, score in ranked]

    @staticmethod
    def _rows(response: Any) -> List[Any]:
        rows = getattr(response, "rows", None)
        if rows is None and isinstance(response, dict):
            rows = response.get("rows")
        return list(rows) if rows else []

    @staticmethod
    def _row_attrs(row: Any) -> Dict[str, Any]:
        """Return a row's attribute dict across SDK row shapes.

        Turbopuffer 2.x returns each row as a pydantic model whose document
        attributes (uri, abstract, ``$dist``, ...) live in ``model_extra`` /
        ``__pydantic_extra__``. Older shapes exposed an ``.attributes`` dict.
        """
        if isinstance(row, dict):
            return dict(row)
        for attr in ("model_extra", "__pydantic_extra__"):
            extra = getattr(row, attr, None)
            if isinstance(extra, dict):
                return dict(extra)
        attrs = getattr(row, "attributes", None)
        if isinstance(attrs, dict):
            return dict(attrs)
        if attrs is not None and hasattr(attrs, "items"):
            return dict(attrs.items())
        return {}

    @staticmethod
    def _row_id(row: Any) -> Any:
        if isinstance(row, dict):
            return row.get("id")
        return getattr(row, "id", None)

    @classmethod
    def _row_fields(cls, row: Any) -> Dict[str, Any]:
        attrs = cls._row_attrs(row)
        # ``vector`` is exposed as a first-class attribute on 2.x rows, fold it in.
        if not isinstance(row, dict):
            vec = getattr(row, "vector", None)
            if vec is not None and "vector" not in attrs:
                attrs["vector"] = vec
        return {k: v for k, v in attrs.items() if k not in ("id", "$dist", "dist")}

    @classmethod
    def _row_score(cls, row: Any) -> Optional[float]:
        attrs = cls._row_attrs(row)
        value = attrs.get("$dist", attrs.get("dist"))
        if value is None and not isinstance(row, dict):
            value = getattr(row, "dist", None)
            if value is None:
                value = getattr(row, "score", None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _row_vector(cls, row: Any) -> Optional[List[float]]:
        if isinstance(row, dict):
            vec = row.get("vector")
        else:
            vec = getattr(row, "vector", None)
            if vec is None:
                vec = cls._row_attrs(row).get("vector")
        if vec is None:
            return None
        try:
            return [float(x) for x in vec]
        except TypeError:
            return None
