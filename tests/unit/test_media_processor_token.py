# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""The per-request git token must reach the accessor but never the parser."""

from types import SimpleNamespace

import pytest

from openviking.parse.accessors.base import LocalResource, SourceType
from openviking.utils.media_processor import UnifiedResourceProcessor


@pytest.mark.asyncio
async def test_token_reaches_accessor_not_parser(tmp_path, monkeypatch):
    f = tmp_path / "readme.md"
    f.write_text("hi")
    captured: dict = {}

    class FakeRegistry:
        async def access(self, source, **kwargs):
            captured["access_kwargs"] = kwargs
            return LocalResource(
                path=f,
                source_type=SourceType.HTTP,
                original_source=source,
                meta={},
                is_temporary=False,
            )

    proc = UnifiedResourceProcessor()
    monkeypatch.setattr(proc, "_get_accessor_registry", lambda: FakeRegistry())
    monkeypatch.setattr(proc, "_get_vlm_processor", lambda: None)

    async def _fake_parse(path, **kwargs):
        captured["parse_kwargs"] = kwargs
        return SimpleNamespace(temp_dir_path=None)

    monkeypatch.setattr("openviking.utils.media_processor.parse", _fake_parse)

    await proc.process("https://github.com/o/r", git_auth_token="TOK")

    # Reaches the accessor...
    assert captured["access_kwargs"].get("git_auth_token") == "TOK"
    # ...but is never forwarded to the parser (which would widen the leak surface).
    assert "git_auth_token" not in captured["parse_kwargs"]
