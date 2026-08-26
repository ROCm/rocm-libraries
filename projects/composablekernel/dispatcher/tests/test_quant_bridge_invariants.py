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

# Bridges with no on-device test.  EMPTY, and it must stay empty: the three
# entries that used to live here (grouped_gemm_{a,ab,b}quant) are now covered by
# tests/test_grouped_quant_gpu_correctness.py, which enumerates every one of
# their shipped default_*_config factories rather than a hand-picked few.
#
# This is the self-expiring form the exemption never had.  There is no date to
# let lapse and no marker to forget: the table is empty, and
# test_i1_bridge_has_on_device_test now fails outright for any bridge that
# loses -- or ships without -- on-device coverage.
_KNOWN_GAPS_NO_GPU_TEST = set()


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

# Bridges not yet on quant_bridge_common.hpp.  Empty: all ten are ported.
_KNOWN_GAPS_NOT_ON_SHARED_LAYER = set()

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


# ---------------------------------------------------------------------------
# Output buffer: C is written in place and must never be copied
# ---------------------------------------------------------------------------
#
# ``C = np.ascontiguousarray(C)`` on the *output* buffer sends the device
# results into a temporary that is discarded on return; the caller's array
# silently keeps its pre-call contents.  Eight of the ten bridges did this.

_QUANT_UTILS = sorted(_PYTHON.glob("*quant*_utils.py"))
_QUANT_UTILS_IDS = [p.name for p in _QUANT_UTILS]

_C_COPY = re.compile(r"^\s*C\s*=\s*np\.ascontiguousarray\(\s*C\s*\)", re.MULTILINE)


@pytest.mark.parametrize("path", _QUANT_UTILS, ids=_QUANT_UTILS_IDS)
def test_output_buffer_is_never_copied(path):
    """No bridge may rebind C to a contiguous copy: results would be discarded."""
    hits = [
        f"{path.name}:{i}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if _C_COPY.match(line)
    ]
    assert not hits, (
        f"{hits} rebinds the output buffer C to a temporary copy; the device "
        f"result is memcpy'd into that temporary and discarded. Validate "
        f"C.flags['C_CONTIGUOUS'] and raise instead."
    )


@pytest.mark.parametrize("path", _QUANT_UTILS, ids=_QUANT_UTILS_IDS)
def test_output_buffer_contiguity_is_validated(path):
    """Each bridge must reject a non-contiguous C rather than silently copying it."""
    text = path.read_text()
    assert 'C.flags["C_CONTIGUOUS"]' in text or "C.flags['C_CONTIGUOUS']" in text, (
        f"{path.name} does not validate that the output buffer C is "
        f"C-contiguous before handing its pointer to the library."
    )


# ---------------------------------------------------------------------------
# Measurement integrity: the instrument must be configurable and fair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _QUANT_UTILS, ids=_QUANT_UTILS_IDS)
def test_every_bridge_compiles_with_te_perf_flags(path):
    """A bridge built with plain -O3 is measured against a TE-flagged baseline.

    That biases every parity number against the bridge, and the repo rule is
    that a >15% gap is a bug, never an artifact -- so an unfair instrument
    manufactures bugs.
    """
    text = path.read_text()
    assert "quant_bridge_flags" in text, (
        f"{path.name} does not import the shared TE perf flags; its .so would "
        f"be built with plain -O3 while Old-TE carries the full -mllvm set."
    )
    assert "_te_perf_flags(" in text or "te_perf_flags(" in text, (
        f"{path.name} imports the TE perf flags but never applies them."
    )


def test_te_flag_strings_are_defined_once():
    """Only quant_bridge_flags may spell the TE -mllvm flag strings."""
    owner = _PYTHON / "quant_bridge_flags.py"
    needles = (
        "-amdgpu-early-inline-all",
        "-amdgpu-function-calls",
        "--lsr-drop-solution",
        "-enable-post-misched",
        "-amdgpu-coerce-illegal-types",
    )
    offenders = []
    for path in _QUANT_UTILS:
        if path == owner:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            for needle in needles:
                if needle in code:
                    offenders.append(f"{path.name}:{lineno}: {needle}")
    assert not offenders, (
        "TE perf flag strings must be defined once in quant_bridge_flags.py; "
        "found copies:\n" + "\n".join(offenders)
    )


def test_so_cache_key_includes_compile_flags():
    """Two different flag sets must not collide on one cached .so filename."""
    import os
    import sys

    sys.path.insert(0, str(_PYTHON))
    import quant_bridge_flags as flags

    flags.coerce_flag_supported.cache_clear()
    previous = os.environ.get("CK_BRIDGE_NO_TE_FLAGS")
    try:
        os.environ["CK_BRIDGE_NO_TE_FLAGS"] = "1"
        without = flags.flags_cache_tag("hipcc")
        os.environ.pop("CK_BRIDGE_NO_TE_FLAGS")
        with_flags = flags.flags_cache_tag("hipcc")
    finally:
        if previous is None:
            os.environ.pop("CK_BRIDGE_NO_TE_FLAGS", None)
        else:
            os.environ["CK_BRIDGE_NO_TE_FLAGS"] = previous
    assert without != with_flags, (
        "CK_BRIDGE_NO_TE_FLAGS does not change the .so cache tag, so flipping "
        "it silently reuses a .so built with the other flag set."
    )

    base = (_PYTHON / "quant_bridge_base.py").read_text()
    assert "flags_tag" in base, (
        "quant_bridge_base does not fold the flag tag into the .so filename."
    )


def test_bquant_flag_order_is_preserved():
    """bquant emits Old-TE's flag order with the coerce flag FIRST."""
    import sys

    sys.path.insert(0, str(_PYTHON))
    import quant_bridge_flags as flags

    expected_tail = [
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        "-mllvm", "--lsr-drop-solution=1",
        "-mllvm", "-enable-post-misched=0",
        "-fno-offload-uniform-block",
        "--offload-compress",
    ]
    original = flags.coerce_flag_supported
    try:
        flags.coerce_flag_supported = lambda _hipcc: False
        assert flags.te_perf_flags(
            "hipcc", extra=("--offload-compress",),
            order=flags.TE_ORDER_BQUANT, coerce_first=True,
        ) == expected_tail

        flags.coerce_flag_supported = lambda _hipcc: True
        assert flags.te_perf_flags(
            "hipcc", extra=("--offload-compress",),
            order=flags.TE_ORDER_BQUANT, coerce_first=True,
        ) == ["-mllvm", "-amdgpu-coerce-illegal-types=1"] + expected_tail
    finally:
        flags.coerce_flag_supported = original
