# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lock handle and LockOwner protocol for path lock integration."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


def _new_lock_id() -> str:
    return str(uuid.uuid4())


@runtime_checkable
class LockOwner(Protocol):
    """Minimal interface that path lock code requires from its caller."""

    id: str
    locks: list[str]
    lock_types: dict[str, str]

    def add_lock(self, path: str, lock_type: Optional[str] = None) -> None:
        raise NotImplementedError

    def remove_lock(self, path: str) -> None:
        raise NotImplementedError


@dataclass
class LockHandle:
    """Identifies a lock holder. Path lock code uses ``id`` to generate fencing tokens
    and ``locks`` to track acquired lock files."""

    id: str = field(default_factory=_new_lock_id)
    locks: list[str] = field(default_factory=list)
    # In-memory record of lock_type per lock_path. Populated at acquire/adopt
    # time so ownership lookups can avoid an S3 GET on every ancestor-walk
    # step — both slow on Express (~100-500ms/read) and race-prone when a
    # parent's lease-refresh task overlaps a consumer's verification read.
    lock_types: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.last_active_at = self.created_at

    def add_lock(self, lock_path: str, lock_type: Optional[str] = None) -> None:
        if lock_path not in self.locks:
            self.locks.append(lock_path)
        if lock_type is not None:
            existing = self.lock_types.get(lock_path)
            # Never downgrade TREE → EXACT. A TREE lock at a given path is
            # strictly more permissive than EXACT (it also covers descendants),
            # so an exact re-acquisition on the same path should not weaken
            # the cached type that `_has_owned_ancestor_tree` reads.
            if existing == "T" and lock_type == "E":
                return
            self.lock_types[lock_path] = lock_type

    def remove_lock(self, lock_path: str) -> None:
        if lock_path in self.locks:
            self.locks.remove(lock_path)
        self.lock_types.pop(lock_path, None)
