# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scoped directed peer mappings using only the existing HIP runtime.

This is host setup/teardown, not a tile transport or a collective implementation.
No library is resolved until a non-empty scope actually queries HIP.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from .hip_module import (
    HipError,
    Runtime,
    _device_ordinal,
    can_access_peer,
    device_scope,
)

__all__ = ["peer_access_scope"]


@contextmanager
def peer_access_scope(paths: Iterable[tuple[int, int]]) -> Iterator[None]:
    """Establish directed ``(execution_device, memory_owner)`` mappings.

    Preflight every capability before enabling any mapping. Existing mappings
    are borrowed, never disabled by this scope. On normal or exceptional exit,
    drain every participating execution device before disabling mappings that
    this scope created. If a drain fails, leave mappings intact and report it.

    Keep allocations/modules alive until scope exit and keep all peer users
    within the scope. The caller must exclusively manage these mappings during
    the scope; concurrent mapping changes or nested overlapping scopes with
    work escaping an inner scope are not supported. This helper is single-node,
    host-thread orchestration, not background communication progress.
    """
    directed = tuple((_device_ordinal(a), _device_ordinal(b)) for a, b in paths)
    if any(a == b for a, b in directed):
        raise ValueError(
            "peer paths must connect different devices; use local views for self"
        )
    if len(set(directed)) != len(directed):
        raise ValueError("peer paths must be unique")
    for execution, owner in directed:
        if not can_access_peer(execution, owner):
            raise HipError(
                f"peer access unsupported from device {execution} to {owner}"
            )

    runtime = Runtime()
    created: list[tuple[int, int]] = []
    entered = False
    try:
        for execution, owner in directed:
            with device_scope(execution):
                if runtime.enable_peer_access(owner):
                    created.append((execution, owner))
        entered = True
        yield
    finally:
        errors: list[Exception] = []
        sources = {a for a, _ in (directed if entered else created)}
        for execution in sorted(sources):
            try:
                with device_scope(execution):
                    runtime.sync()
            except (HipError, OSError, AttributeError) as error:
                errors.append(error)
        if errors:
            raise HipError(
                "peer scope drain failed; mappings remain enabled"
            ) from errors[0]
        for execution, owner in reversed(created):
            try:
                with device_scope(execution):
                    runtime.disable_peer_access(owner)
            except (HipError, OSError, AttributeError) as error:
                errors.append(error)
        if errors:
            raise HipError("peer scope mapping cleanup failed") from errors[0]
