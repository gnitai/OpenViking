# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Turbopuffer backend collection adapter."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openviking.storage.expr import (
    And,
    Contains,
    Eq,
    FilterExpr,
    In,
    Or,
    PathScope,
    Range,
    RawDSL,
    TimeRange,
)
from openviking.storage.vectordb.collection.collection import Collection
from openviking.storage.vectordb.collection.turbopuffer_collection import (
    _build_client,
    get_or_create_turbopuffer_collection,
)
from openviking_cli.utils import get_logger

from .base import CollectionAdapter

logger = get_logger(__name__)


_DISTANCE_MAP = {
    "cosine": "cosine_distance",
    "cosine_distance": "cosine_distance",
    "l2": "euclidean_squared",
    "euclidean": "euclidean_squared",
    "euclidean_squared": "euclidean_squared",
}


def _map_distance(distance: str) -> str:
    key = (distance or "cosine").lower()
    mapped = _DISTANCE_MAP.get(key)
    if mapped is None:
        raise ValueError(
            f"Turbopuffer does not support distance metric '{distance}'. Use one of: cosine, l2."
        )
    return mapped


class TurbopufferCollectionAdapter(CollectionAdapter):
    """Adapter for the managed Turbopuffer vector DB."""

    def __init__(
        self,
        *,
        api_key: str,
        region: Optional[str],
        base_url: Optional[str],
        namespace_name: str,
        collection_name: str,
        index_name: str,
        bm25_field: str = "text",
    ):
        super().__init__(collection_name=collection_name, index_name=index_name)
        self.mode = "turbopuffer"
        self._api_key = api_key
        self._region = region
        self._base_url = base_url
        self._namespace_name = namespace_name
        self._bm25_field = bm25_field
        self._client: Optional[Any] = None
        self._pending_distance: Optional[str] = None
        self._pending_schema: Optional[Dict[str, Any]] = None
        self._pending_fields: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: Any) -> "TurbopufferCollectionAdapter":
        tp = getattr(config, "turbopuffer", None)
        api_key = (tp.api_key if tp else None) or os.environ.get("TURBOPUFFER_API_KEY")
        if not api_key:
            raise ValueError(
                "Turbopuffer backend requires 'api_key' (or TURBOPUFFER_API_KEY env var)"
            )
        region = tp.region if tp else None
        base_url = tp.base_url if tp else None
        if not region and not base_url:
            raise ValueError("Turbopuffer backend requires 'region' or 'base_url' to be set")
        collection_name = config.name or "context"
        namespace_name = (tp.namespace if tp and tp.namespace else None) or collection_name
        bm25_field = tp.bm25_field if tp and tp.bm25_field else "text"
        return cls(
            api_key=api_key,
            region=region,
            base_url=base_url,
            namespace_name=namespace_name,
            collection_name=collection_name,
            index_name=config.index_name or "default",
            bm25_field=bm25_field,
        )

    # ----------------------------------------------------------- client mgmt

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _build_client(
                api_key=self._api_key,
                region=self._region,
                base_url=self._base_url,
            )
        return self._client

    def _remote_has_namespace(self) -> bool:
        client = self._get_client()
        try:
            for summary in client.namespaces(prefix=self._namespace_name):
                if getattr(summary, "id", None) == self._namespace_name:
                    return True
        except Exception:
            logger.exception(
                "Failed to list turbopuffer namespaces (prefix=%s)",
                self._namespace_name,
            )
        return False

    def _wrap_collection(self) -> Collection:
        # Surface the OV field definitions through the collection's meta_data.
        # Callers gate writes on the collection's known fields (see
        # _SingleAccountBackend._filter_known_fields); without "Fields" here the
        # allow-set is empty and every attribute — including "vector" — is stripped.
        meta_data: Dict[str, Any] = {"CollectionName": self._collection_name}
        if self._pending_fields:
            meta_data["Fields"] = list(self._pending_fields)
        return get_or_create_turbopuffer_collection(
            client=self._get_client(),
            namespace_name=self._namespace_name,
            distance_metric=self._pending_distance,
            schema=self._pending_schema,
            bm25_field=self._bm25_field,
            meta_data=meta_data,
        )

    def _load_existing_collection_if_needed(self) -> None:
        if self._collection is not None:
            return
        if not self._remote_has_namespace():
            return
        self._collection = self._wrap_collection()

    def _create_backend_collection(self, meta: Dict[str, Any]) -> Collection:
        # Capture the OV field definitions so _build_default_index_meta can
        # translate them into an explicit Turbopuffer schema (see _translate_schema).
        self._pending_fields = list(meta.get("Fields", []) or [])
        return self._wrap_collection()

    # ----------------------------------------------------------- index shape

    def _translate_schema(
        self, fields: List[Dict[str, Any]], tp_distance: str
    ) -> Dict[str, Any]:
        """Translate the OV collection ``Fields`` into a Turbopuffer schema.

        Turbopuffer infers untyped attributes from the first written value, which
        mis-types whole-number floats as ``int`` and never declares the vector
        index distance metric. Declaring types up front avoids those failures.
        """
        schema: Dict[str, Any] = {}
        for field in fields:
            name = field.get("FieldName")
            ftype = (field.get("FieldType") or "").lower()
            if not name:
                continue
            # ``id`` is Turbopuffer's implicit primary key; it is not a declarable attribute.
            if field.get("IsPrimaryKey") or name == "id":
                continue

            if ftype == "vector":
                dim = field.get("Dim")
                if not dim:
                    continue
                schema[name] = {
                    "type": f"[{int(dim)}]f32",
                    "ann": {"distance_metric": tp_distance},
                }
            elif ftype == "sparse_vector":
                schema[name] = {
                    "type": "{}f16",
                    "sparse_knn": {"distance_metric": "dot_product"},
                }
            elif ftype == "int64":
                schema[name] = "int"
            elif ftype in ("float", "float32"):
                schema[name] = "float"
            elif ftype == "bool":
                schema[name] = "bool"
            elif ftype == "path":
                # uri/parent_uri are filtered with Glob/PathScope → enable glob.
                schema[name] = {"type": "string", "filterable": True, "glob": True}
            elif ftype == "text":
                schema[name] = {"type": "string", "full_text_search": True}
            elif ftype == "string":
                if name == self._bm25_field:
                    schema[name] = {"type": "string", "full_text_search": True}
                else:
                    schema[name] = "string"
            elif ftype == "date_time":
                # OV writes timestamps as ISO8601 strings, not epochs.
                schema[name] = "string"
            elif ftype == "list<string>":
                schema[name] = "[]string"
            elif ftype == "list<int64>":
                schema[name] = "[]int"
            # geo_point / unknown types: leave undeclared and let TP infer.
        return schema

    def _build_default_index_meta(
        self,
        *,
        index_name: str,
        distance: str,
        use_sparse: bool,
        sparse_weight: float,
        scalar_index_fields: List[str],
    ) -> Dict[str, Any]:
        tp_distance = _map_distance(distance)
        self._pending_distance = tp_distance
        schema: Dict[str, Any] = self._translate_schema(self._pending_fields, tp_distance)
        self._pending_schema = schema
        # Record scalar-index fields so callers can introspect; TP indexes by default.
        index_meta: Dict[str, Any] = {
            "IndexName": index_name,
            "distance_metric": tp_distance,
            "schema": schema,
            "scalar_index_fields": list(scalar_index_fields),
        }
        if use_sparse:
            index_meta["sparse_weight"] = sparse_weight
        return index_meta

    # ----------------------------------------------------- filter compilation

    def _compile_filter(self, expr: FilterExpr | Dict[str, Any] | None) -> Any:
        """Compile a FilterExpr into Turbopuffer's tuple-shaped filter form."""
        if expr is None:
            return None
        if isinstance(expr, dict):
            # Caller passed a raw TP-shaped filter; pass through.
            return expr
        if isinstance(expr, RawDSL):
            return expr.payload
        if isinstance(expr, And):
            conds = [c for c in (self._compile_filter(c) for c in expr.conds) if c]
            if not conds:
                return None
            if len(conds) == 1:
                return conds[0]
            return ("And", conds)
        if isinstance(expr, Or):
            conds = [c for c in (self._compile_filter(c) for c in expr.conds) if c]
            if not conds:
                return None
            if len(conds) == 1:
                return conds[0]
            return ("Or", conds)
        if isinstance(expr, Eq):
            value = self._encode_field_value(expr.field, expr.value)
            return (expr.field, "Eq", value)
        if isinstance(expr, In):
            values = [self._encode_field_value(expr.field, v) for v in expr.values]
            return (expr.field, "In", values)
        if isinstance(expr, Range):
            parts: List[Any] = []
            if expr.gte is not None:
                parts.append((expr.field, "Gte", expr.gte))
            if expr.gt is not None:
                parts.append((expr.field, "Gt", expr.gt))
            if expr.lte is not None:
                parts.append((expr.field, "Lte", expr.lte))
            if expr.lt is not None:
                parts.append((expr.field, "Lt", expr.lt))
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return ("And", parts)
        if isinstance(expr, TimeRange):
            parts = []
            if expr.start is not None:
                parts.append((expr.field, "Gte", expr.start))
            if expr.end is not None:
                parts.append((expr.field, "Lt", expr.end))
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return ("And", parts)
        if isinstance(expr, Contains):
            return (expr.field, "Glob", f"*{expr.substring}*")
        if isinstance(expr, PathScope):
            encoded = self._encode_field_value(expr.field, expr.path)
            prefix = encoded.rstrip("/") if isinstance(encoded, str) else encoded
            if not isinstance(prefix, str):
                return (expr.field, "Eq", encoded)
            if expr.depth <= 0:
                return (expr.field, "Glob", f"{prefix}/*")
            # Bounded depth: match prefix plus exactly `depth` more segments.
            suffix = "/*" * expr.depth
            return (expr.field, "Glob", f"{prefix}{suffix}")
        raise TypeError(f"Unsupported filter expr type: {type(expr)!r}")

    def _encode_field_value(self, field: str, value: Any) -> Any:
        if field in self._URI_FIELD_NAMES and isinstance(value, str):
            return self._encode_uri_field_value(value)
        return value
