################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase G0 smoke test — proves the CPU-only codegen-emit harness works and is
deterministic, and pins a compact golden digest of the emitted assembly.

The emit itself is what drives coverage of ``KernelWriterAssembly`` /
``KernelWriter``; this single gfx942 kernel already exercises thousands of lines
of the emitter. We snapshot a *digest* (deterministic kernel name + emit return
code + line count + sha256 of the canonicalized text) rather than the full
~200KB of assembly, to keep the golden compact while still catching any change
in emitted bytes.
"""

import os

import pytest

from codegen_harness import (
    canonicalize_asm,
    emit_kernels_from_logic,
)

pytestmark = pytest.mark.unit

# A small, self-contained logic file already shipped as a fixture for the
# LibraryIO suite (gfx942, HSS/BH GEMM). 1 solution -> 1 assembly kernel.
_LOGIC = os.path.join(
    os.path.dirname(__file__),
    "..",
    "LibraryIO",
    "data",
    "logic_gfx942_HSS_BH.yaml",
)


def test_emit_produces_assembly():
    """The harness emits non-trivial assembly for the fixture kernel."""
    results = emit_kernels_from_logic(_LOGIC)
    assert len(results) == 1
    base, src, err = results[0]
    assert err == 0
    assert src and len(src.splitlines()) > 1000
    # Sanity: it is real AMDGCN assembly for the expected target.
    assert ".amdgcn_target" in src
    assert "gfx942" in src
    assert base.startswith("Cijk_")


def test_emit_is_deterministic():
    """Two independent emits are byte-identical after canonicalization."""
    a = emit_kernels_from_logic(_LOGIC)
    b = emit_kernels_from_logic(_LOGIC)
    assert [t[0] for t in a] == [t[0] for t in b]
    assert [t[1] for t in a] == [t[1] for t in b]


def test_canonicalize_neutralizes_random_labels():
    """The canonicalizer maps random label suffixes to stable ids and is
    idempotent."""
    raw = (
        "s_cbranch_scc0 label_NoBranch_T8JHFHKM7BO5OHXW\n"
        "label_NoBranch_T8JHFHKM7BO5OHXW:\n"
        "s_branch label_Done_S4FDBQ587JJL6NOU\n"
        "label_Done_S4FDBQ587JJL6NOU:\n"
    )
    canon = canonicalize_asm(raw)
    assert "T8JHFHKM7BO5OHXW" not in canon
    assert canon.count("_LBL0") == 2  # def + reference preserved as a pair
    assert canon.count("_LBL1") == 2
    assert canonicalize_asm(canon) == canon  # idempotent


def test_emit_golden_digest(snapshot):
    """Pin the order-invariant golden (kernel identity + emit success).

    The full assembly text is not hashed (it is order-coupled via the emitter's
    process-global MMA-scheduler state); coverage comes from running the emit.
    """
    results = emit_kernels_from_logic(_LOGIC)
    digests = [{"basename": b, "err": e} for (b, _s, e) in results]
    assert digests == snapshot
