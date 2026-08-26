#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mechanical invariants that stop the quant-bridge copy-paste pattern replicating.

Every previous attempt to hold this line was social -- an owner, a reviewer of
record, a follow-up ticket -- and every one of them lost: two more copy-paste
grouped bridges landed on ``develop`` while the consolidation of the first eight
was under review.  These three invariants are the mechanical replacement.  They
are CPU-only, need no GPU and no compiler, and they fail the build the moment an
eleventh bridge arrives without its coverage.

  I1  every ctypes quant bridge has an on-device test that imports its utils;
  I2  every ``tests/test_*.py`` is registered in CTest;
  I3  every ctypes quant bridge includes ``quant_bridge_common.hpp``.

Bridges that violate an invariant *today* are listed in the ``_KNOWN_GAPS``
tables below with the work item that closes them.  Those tables are themselves
asserted to be **minimal**: an entry that has stopped being a violation is a
failure, so a gap cannot silently outlive its fix, and adding a new entry is a
visible, reviewable diff rather than a test that quietly never ran.
"""

import re
from pathlib import Path

import pytest

_DISP = Path(__file__).resolve().parent.parent
_TESTS = _DISP / "tests"
_CTYPES = _DISP / "bindings" / "ctypes"
_PYTHON = _DISP / "python"
_CMAKE = _TESTS / "CMakeLists.txt"


def _quant_bridges():
    """(op, source path) for every ctypes quant bridge, e.g. ``grouped_gemm_aquant``."""
    out = []
    for path in sorted(_CTYPES.glob("*quant*_ctypes_lib.cpp")):
        out.append((path.name[: -len("_ctypes_lib.cpp")], path))
    return out


_BRIDGES = _quant_bridges()
_BRIDGE_IDS = [op for op, _ in _BRIDGES]


def test_bridge_discovery_is_not_empty():
    """A typo in the glob would make every invariant below vacuously true."""
    assert len(_BRIDGES) >= 10, _BRIDGE_IDS


# ---------------------------------------------------------------------------
# I1 -- on-device coverage
# ---------------------------------------------------------------------------

# Bridges with no on-device test.  Closing these is the "grouped-bquant lost its
# only on-device coverage" work item; each needs a *_gpu_correctness.py that
# imports the op's utils module.
_KNOWN_GAPS_NO_GPU_TEST = {
    "grouped_gemm_aquant",
    "grouped_gemm_abquant",
    "grouped_gemm_bquant",
}


def _gpu_tests_importing(utils_module: str):
    """GPU-correctness test files that import ``utils_module`` as a whole module."""
    pattern = re.compile(
        rf"^\s*(?:from\s+{re.escape(utils_module)}\s+import|import\s+{re.escape(utils_module)})\b",
        re.MULTILINE,
    )
    return [
        p.name
        for p in sorted(_TESTS.glob("test_*gpu_correctness*.py"))
        if pattern.search(p.read_text())
    ]


@pytest.mark.parametrize("op,src", _BRIDGES, ids=_BRIDGE_IDS)
def test_i1_bridge_has_on_device_test(op, src):
    """Every ctypes quant bridge must have an on-device test importing its utils."""
    utils_module = f"{op}_utils"
    assert (_PYTHON / f"{utils_module}.py").exists(), (
        f"{src.name} has no {utils_module}.py"
    )
    found = _gpu_tests_importing(utils_module)
    if op in _KNOWN_GAPS_NO_GPU_TEST:
        assert not found, (
            f"{op} now has on-device coverage ({found}); remove it from "
            f"_KNOWN_GAPS_NO_GPU_TEST so the invariant starts enforcing it"
        )
        pytest.skip(f"{op}: known gap, no on-device test yet")
    assert found, (
        f"{src.name} has no on-device test importing {utils_module}. "
        f"Add tests/test_{op}_gpu_correctness.py, or -- only with a reason -- "
        f"add {op!r} to _KNOWN_GAPS_NO_GPU_TEST."
    )


# ---------------------------------------------------------------------------
# I2 -- CTest registration
# ---------------------------------------------------------------------------

# Test files that CTest does not invoke today.  All of them predate the quant
# bridges and belong to other operators; they are listed rather than fixed here
# so that this invariant still refuses **new** unregistered test files.  The set
# is asserted to be minimal below, so registering one forces deleting its entry.
_UNREGISTERED_LEGACY = {
    "test_batched_bridge.py",
    "test_batched_contraction_bridge.py",
    "test_codegen_common.py",
    "test_depthwise_tile_math.py",
    "test_dispatcher_common.py",
    "test_gemm_parity.py",
    "test_grouped_conv_codegen.py",
    "test_grouped_conv_utils.py",
    "test_grouped_gemm_codegen.py",
    "test_library_caching.py",
    "test_multi_abd_bridge.py",
    "test_multi_d_bridge.py",
    "test_mx_gemm_bridge.py",
    "test_rules_coverage.py",
    "test_streamk_gemm_utils.py",
    "test_tile_math.py",
}

_FOREACH = re.compile(r"foreach\((\w+)\s+([^)]*)\)(.*?)endforeach\(\)", re.S)


def _ctest_text() -> str:
    """CMakeLists text with ``foreach`` bodies expanded once per loop item.

    Most suites are registered through a ``foreach(_bs_op ...)`` style loop, so
    a plain substring search would report them as missing.
    """
    text = _CMAKE.read_text()
    parts = [text]
    for match in _FOREACH.finditer(text):
        var, items, body = match.group(1), match.group(2).split(), match.group(3)
        for item in items:
            parts.append(body.replace("${" + var + "}", item))
    return "\n".join(parts)


_ALL_TEST_FILES = sorted(p.name for p in _TESTS.glob("test_*.py"))


@pytest.mark.parametrize("test_file", _ALL_TEST_FILES)
def test_i2_every_test_file_is_registered(test_file):
    """A test file that CTest never invokes is a test that has never run."""
    stem = test_file[: -len(".py")]
    registered = stem in _ctest_text()
    if test_file in _UNREGISTERED_LEGACY:
        assert not registered, (
            f"{test_file} is now registered; remove it from "
            f"_UNREGISTERED_LEGACY so the invariant starts enforcing it"
        )
        pytest.skip(f"{test_file}: known-unregistered legacy suite")
    assert registered, (
        f"{test_file} is not referenced in tests/CMakeLists.txt, so CI has "
        f"never executed it. Register it (the _quant_test foreach is the usual "
        f"home for CPU suites)."
    )


def test_i2_legacy_exemptions_all_exist():
    """A stale exemption is an invariant that silently stopped being enforced."""
    missing = sorted(_UNREGISTERED_LEGACY - set(_ALL_TEST_FILES))
    assert not missing, f"_UNREGISTERED_LEGACY names deleted files: {missing}"


# ---------------------------------------------------------------------------
# I3 -- shared-layer adoption
# ---------------------------------------------------------------------------

# Bridges not yet on quant_bridge_common.hpp.  These are the two that landed on
# develop in the old copy-paste style; porting them is the "port the two
# develop-landed bridges onto the shared layer" work item.
_KNOWN_GAPS_NOT_ON_SHARED_LAYER = {
    "grouped_gemm_rowcolquant",
    "grouped_gemm_tensorquant",
}

_SHARED_HEADER = "quant_bridge_common.hpp"


@pytest.mark.parametrize("op,src", _BRIDGES, ids=_BRIDGE_IDS)
def test_i3_bridge_is_on_shared_layer(op, src):
    """A new quant bridge that re-implements alloc/timing/cleanup must fail CI."""
    adopted = _SHARED_HEADER in src.read_text()
    if op in _KNOWN_GAPS_NOT_ON_SHARED_LAYER:
        assert not adopted, (
            f"{op} now includes {_SHARED_HEADER}; remove it from "
            f"_KNOWN_GAPS_NOT_ON_SHARED_LAYER so the invariant starts enforcing it"
        )
        pytest.skip(f"{op}: known gap, not yet ported to the shared layer")
    assert adopted, (
        f"{src.name} does not include {_SHARED_HEADER}. Every quant bridge "
        f"shares its allocation, arch validation, timing, error handling and "
        f"cleanup; re-implementing them is how the last ten diverged."
    )


# ---------------------------------------------------------------------------
# The exemption tables must stay minimal and must name real bridges.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table_name,table",
    [
        ("_KNOWN_GAPS_NO_GPU_TEST", _KNOWN_GAPS_NO_GPU_TEST),
        ("_KNOWN_GAPS_NOT_ON_SHARED_LAYER", _KNOWN_GAPS_NOT_ON_SHARED_LAYER),
    ],
)
def test_known_gap_tables_name_real_bridges(table_name, table):
    """A stale exemption is an invariant that silently stopped being enforced."""
    unknown = sorted(table - set(_BRIDGE_IDS))
    assert not unknown, (
        f"{table_name} names bridges that no longer exist: {unknown}"
    )
