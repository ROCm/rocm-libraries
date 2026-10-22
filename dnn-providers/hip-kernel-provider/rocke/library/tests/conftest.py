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

import os
import sys
from pathlib import Path

import pytest

_LIBROOT = Path(__file__).resolve().parents[1]  # tests -> rocke/library
# Keep the library packages ahead of the CTest working directory. In the
# standalone install layout that directory is bin/rocke, whose platform
# `dispatch` subpackage would otherwise shadow the library's top-level package.
try:
    sys.path.remove(str(_LIBROOT))
except ValueError:
    pass
sys.path.insert(0, str(_LIBROOT))

_PYROOT = Path(__file__).resolve().parents[2] / "platform" / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))

# Test support packages are also used by standalone worker subprocesses. Make
# them importable with pytest's importlib mode in both source and install trees,
# after the library packages so tests/dispatch cannot shadow dispatch/.
_TESTROOT = Path(__file__).resolve().parent
if str(_TESTROOT) not in sys.path:
    sys.path.append(str(_TESTROOT))

# pytest rewrites asserts only in test modules and conftest files. Every
# operation's pinned GPU tests come from this helper, so register it before a
# test module imports it; otherwise a failed assert shows none of its values.
pytest.register_assert_rewrite("reference_common.pinned_tests")


def pytest_addoption(parser):
    from reference_common.artifact import OPERATIONS

    parser.addoption("--rocke-reference-arch", default=None)
    parser.addoption("--rocke-reference-bundle", type=Path, default=None)
    parser.addoption("--rocke-reference-lock", type=Path, default=None)
    parser.addoption("--rocke-reference-operation", choices=OPERATIONS, default=None)
    parser.addoption(
        "--gdn-batch",
        action="store",
        type=int,
        default=None,
        help="run GDN all-candidate numeric coverage for one batch",
    )
    parser.addoption(
        "--gdn-spec-id",
        action="store",
        default=None,
        help="run GDN all-candidate numeric coverage for one stable spec ID",
    )


def pytest_configure(config):
    """Pin the production comgr (the torch wheel's bundled library) by path
    for this session and its child processes, without importing torch
    (``kernels/common/attention_bwd_comgr.py``), and give comgr's code-object
    cache a directory of this session's own (removed at exit). An explicit ``ROCKE_COMGR_LIB`` /
    ``AMD_COMGR_CACHE_DIR`` wins; ``ROCKE_BWD_NO_COMGR_PIN=1`` opts out.

    An import failure of the ``kernels`` package is reported, not raised: the
    tests that need the package fail on it themselves, the others still run
    (the same rule as ``_restore_attention_arch_state`` below)."""
    try:
        from kernels.common.attention_bwd_comgr import ensure_production_comgr
    except Exception as exc:  # noqa: BLE001 - see the docstring
        config.issue_config_time_warning(
            pytest.PytestConfigWarning(f"production comgr not pinned: {exc!r}"),
            stacklevel=2,
        )
        return
    ensure_production_comgr()


def pytest_collection_finish(session):
    """``ROCKE_BWD_PREBUILD=1``: compile the selected backward tests' kernels
    once into ``ROCKE_BWD_HSACO_CACHE`` (``tests/_attention_bwd_prebuild.py``)."""
    if os.environ.get("ROCKE_BWD_PREBUILD", "").strip() not in ("", "0"):
        from tests._attention_bwd_prebuild import prebuild_session

        prebuild_session(session)


def pytest_terminal_summary(terminalreporter):
    """With ``ROCKE_BWD_HSACO_CACHE``: one ``bwd-hsaco-cache:`` line of this
    session's disk-cache counters (``misses`` = kernels compiled here)."""
    ac = sys.modules.get("kernels.common.attention_bwd_cache")
    if ac is None or not os.environ.get("ROCKE_BWD_HSACO_CACHE", "").strip():
        return
    stats = ac.env_cache_stats()
    terminalreporter.write_line(
        "bwd-hsaco-cache: " + " ".join(f"{k}={v}" for k, v in stats.items())
    )


@pytest.fixture(autouse=True)
def _restore_attention_arch_state():
    """Undo any write to the process-wide arch memo and its derived cache.

    ``attention_unified._RESOLVED_ATTENTION_ARCH`` memoizes the real device arch
    on first resolve, and a dozen tests pin it directly to exercise a specific
    arch. One that leaves it set -- or cleared -- changes what every later test
    resolves, which surfaces as order-dependent failures on a GPU runner and not
    at all on a CPU one. Restoring it here makes the leak class impossible
    rather than relying on each test's own teardown.

    ``_2D_LAUNCH_META`` and ``_2D_KV_ADDR_LIMITS`` are *cleared*, not restored:
    they are pure memos whose entries are derived from whatever arch was
    resolved when they were computed, so a pinned-gfx942 entry must not be
    served back to a gfx950 test. ``test_attn_k_slice_hd`` and
    ``test_attn_bf16_d128_ring`` clear the launch meta by hand today; this makes
    that unnecessary.

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
            au._2D_KV_ADDR_LIMITS.clear()


def pytest_sessionstart(session):
    from reference_common.pytest_support import start_reference_session

    start_reference_session(session.config)


def pytest_collection_modifyitems(config, items):
    """``--rocke-reference-arch``: keep that arch's reference cases; then
    ``ROCKE_TEST_SHARD=i[,j...]/n``: keep the items of those shards (see
    ``tests/run_device_shards.py``; unset: every item)."""
    from reference_common.pytest_support import select_reference_items

    select_reference_items(config, items)
    if os.environ.get("ROCKE_TEST_SHARD", "").strip():
        from tests.run_device_shards import select_items

        select_items(config, items)
