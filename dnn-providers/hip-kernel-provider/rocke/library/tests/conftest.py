# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Pytest root config for the rocKE library test tree. Puts BOTH the library
# source root (rocke/library, exposing `kernels`/`builders`/`dispatch`) and the
# platform Python engine root (rocke/platform/python, exposing `rocke`) on
# sys.path so the attention tests resolve without an external PYTHONPATH. The
# library legally depends on the platform SDK (one-way rule: library -> platform);
# the reverse is forbidden. Paths are derived from this file's location so the
# tree stays copy-able verbatim into another repo.
#
# parents[1] -> rocke/library
# parents[2] -> rocke

import sys
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]  # tests -> rocke/library
if str(_LIBROOT) not in sys.path:
    sys.path.insert(0, str(_LIBROOT))

_PYROOT = Path(__file__).resolve().parents[2] / "platform" / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))


@pytest.fixture(autouse=True)
def _restore_attention_arch_state():
    """Undo any write to the process-wide arch memo and its derived cache.

    ``attention_unified._RESOLVED_ATTENTION_ARCH`` memoizes the real device arch
    on first resolve, and a dozen tests pin it directly to exercise a specific
    arch. One that leaves it set -- or cleared -- changes what every later test
    resolves, which surfaces as order-dependent failures on a GPU runner and not
    at all on a CPU one. Restoring it here makes the leak class impossible
    rather than relying on each test's own teardown.

    ``_2D_LAUNCH_META`` is *cleared*, not restored: it is a pure memo whose
    entries are derived from whatever arch was resolved when they were computed,
    so a pinned-gfx942 entry must not be served back to a gfx950 test.
    ``test_attn_k_slice_hd`` and ``test_attn_bf16_d128_ring`` clear it by hand
    today; this makes that unnecessary.

    Deliberately NOT covered here: ``HIPDNN_GFX942_*`` env overrides. An env leak
    belongs to the test that pops them (``monkeypatch.delenv``), not to a blanket
    conftest snapshot -- a snapshot would mask the leak rather than fix it.
    """
    # Looked up, not imported: this fixture runs for every test in the tree,
    # including ones with nothing to do with attention. Forcing the import here
    # would turn any import regression in ``attention_unified`` into a tree-wide
    # setup error. A test that never imported the module cannot have written the
    # memo, and ``None`` is the module's pristine value, so the not-imported
    # branch restores correctly either way.
    _MOD = "kernels.common.attention_unified"
    au = sys.modules.get(_MOD)
    prev = au._RESOLVED_ATTENTION_ARCH if au is not None else None
    try:
        yield
    finally:
        au = sys.modules.get(_MOD)
        if au is not None:
            au._RESOLVED_ATTENTION_ARCH = prev
            au._2D_LAUNCH_META.clear()
