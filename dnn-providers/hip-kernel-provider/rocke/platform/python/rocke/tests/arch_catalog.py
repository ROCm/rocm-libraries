# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: serve a synthetic arch catalog, leaving no trace of it behind.

Several lookups in ``rocke.core.arch.target`` are ``lru_cache``d on the loaded
catalog, and some are filled as a side effect of other calls (building an IR
``mma`` op resolves its family, physical atom and destination type). A test that
patches ``_load_specs`` and clears only the cache it has in mind leaves the
others holding its synthetic catalog, and whichever test runs next in the
process reads it: a later test then passes or fails depending on which tests
ran before it.

    with patched_arch_specs(drifted):
        with pytest.raises(ValueError, match="inconsistent family"):
            _op_id_family()
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from unittest import mock

from ..core.arch import target

__all__ = ["patched_arch_specs"]


def _clear_catalog_caches() -> None:
    for value in vars(target).values():
        if hasattr(value, "cache_clear"):
            value.cache_clear()


@contextlib.contextmanager
def patched_arch_specs(specs: Mapping[str, dict]) -> Iterator[None]:
    """Serve ``specs`` as the arch catalog inside the block. Every catalog cache
    in ``rocke.core.arch.target`` is cleared on entry, so the block reads only
    ``specs``, and on exit, so nothing after it reads ``specs``."""
    _clear_catalog_caches()
    try:
        with mock.patch.object(target, "_load_specs", return_value=specs):
            yield
    finally:
        _clear_catalog_caches()
