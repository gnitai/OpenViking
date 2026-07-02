# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""The two sidecar files (.overview.md / .abstract.md) must be written
concurrently, not one after the other — they are independent objects and on
S3 Express the per-write latency dominates."""

import asyncio

import pytest

from openviking.storage.queuefs.semantic_sidecar import _write_sidecars


class _PeakFS:
    """Records the peak number of write_file calls in flight at once."""

    def __init__(self):
        self.inflight = 0
        self.peak = 0
        self.writes = {}

    async def write_file(self, path, content, ctx=None):
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        # Yield so a concurrently-scheduled write can interleave. If the two
        # writes are awaited sequentially, the second one hasn't even started
        # when the first runs, so peak stays at 1.
        await asyncio.sleep(0)
        self.inflight -= 1
        self.writes[path] = content


@pytest.mark.asyncio
async def test_write_sidecars_issues_both_writes_concurrently():
    fs = _PeakFS()

    await _write_sidecars(fs, "wfs://x/dir", "OV", "AB", None)

    assert fs.writes == {
        "wfs://x/dir/.overview.md": "OV",
        "wfs://x/dir/.abstract.md": "AB",
    }
    assert fs.peak == 2, (
        f"expected both sidecar writes in flight at once, peak was {fs.peak} "
        "(writes are still sequential)"
    )


if __name__ == "__main__":
    pytest.main([__file__])
