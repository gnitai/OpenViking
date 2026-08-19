#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Stamp the legacy embedding identity onto every pre-existing project.

RUN THIS BEFORE CHANGING ``embedding.dense.model`` IN THE CONFIG.

Projects created before per-project pinning have no pin on disk. They resolve
through the frozen ``embedding.legacy`` block, which is correct -- but relying
on that fallback fleet-wide means one bad edit to the legacy block silently
moves every old project into a vector space its data is not in. Writing a real
pin per project removes the ambiguity: after this runs, an absent pin means
"genuinely new project", nothing else.

Idempotent: a project that already has a pin is left untouched, so re-running
after a partial failure is safe.

Usage:
    python scripts/backfill_embedding_pins.py --dry-run
    python scripts/backfill_embedding_pins.py
    python scripts/backfill_embedding_pins.py --accounts-file ids.txt
"""

import argparse
import asyncio
import sys
from typing import List, Optional

from openviking.models.embedder.identity import legacy_identity
from openviking.storage.project_embedding import (
    PIN_PATH_TEMPLATE,
    ProjectEmbeddingPins,
)


async def _discover_accounts(pins: ProjectEmbeddingPins) -> List[str]:
    """List account ids from the AGFS root (/local/<account_id>/...)."""
    entries = await pins._async_agfs.ls("/local")
    accounts = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name or name == "_system":
            continue
        accounts.append(name)
    return sorted(accounts)


def _read_accounts_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


async def _run(args: argparse.Namespace) -> int:
    from openviking.service.core import OpenVikingService
    from openviking_cli.utils.config import get_openviking_config

    # The service constructor loads ov.conf, so build it before reading config.
    service = OpenVikingService()
    await service.initialize()

    identity = legacy_identity(get_openviking_config())
    print(f"Legacy identity to stamp: {identity.to_dict()}")

    viking_fs = service.viking_fs
    if viking_fs is None:
        print("ERROR: VikingFS unavailable; cannot reach per-project storage", file=sys.stderr)
        return 1
    pins = ProjectEmbeddingPins(viking_fs)

    if args.accounts_file:
        accounts = _read_accounts_file(args.accounts_file)
    else:
        accounts = await _discover_accounts(pins)
    print(f"Found {len(accounts)} account(s)")

    stamped = skipped = failed = 0
    for account_id in accounts:
        existing = await pins._read_pin(account_id)
        if existing is not None:
            skipped += 1
            print(f"  skip   {account_id}: already pinned to {existing.model}")
            continue
        if args.dry_run:
            stamped += 1
            print(f"  WOULD  {account_id} -> {identity.model}")
            continue
        try:
            await pins._write_pin(account_id, identity)
        except Exception as err:  # noqa: BLE001 - report and continue the sweep
            failed += 1
            print(f"  FAIL   {account_id}: {err}", file=sys.stderr)
            continue
        stamped += 1
        print(f"  pinned {account_id} -> {identity.model}")

    verb = "would stamp" if args.dry_run else "stamped"
    print(f"\n{verb}={stamped} skipped={skipped} failed={failed}")
    print(f"Pin path template: {PIN_PATH_TEMPLATE}")
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be stamped without writing anything",
    )
    parser.add_argument(
        "--accounts-file",
        help=(
            "file of account ids, one per line, instead of discovering them "
            "from AGFS (e.g. project ids exported from the workik context table)"
        ),
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
