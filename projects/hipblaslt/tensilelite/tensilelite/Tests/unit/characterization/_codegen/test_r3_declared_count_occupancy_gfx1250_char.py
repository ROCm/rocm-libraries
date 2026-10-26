# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 CUOccupancy must follow the emitted descriptor, not the VGPR pool.

``checkResources`` judges a kernel before StinkyTofu has emitted it, so the pool
sizes it sees are a proposal.  ``emitAssembly`` then rewrites
``.amdhsa_next_free_vgpr`` / ``_sgpr`` to what the final code uses -- at every
``StinkyTofuRegisterAllocation`` mode including the default 0, because the
refresh is part of emitting, not part of allocating -- and the hardware reserves
what the descriptor says.  ``resolveDeferredRegisterOverflow`` therefore judges
the budget again on those counts and recomputes ``CUOccupancy`` from them.

Why this needs a test on real kernels rather than only the unit tests in
``Tests/unit/test_register_overflow_deferral.py``: ``CUOccupancy`` leaves codegen
and becomes ``sizeMapping.CUOccupancy``, which ``ContractionSolution.cpp`` uses
for Stream-K grid sizing and the CU-count heuristics.  No characterization
snapshot records it -- the ``CUOccupancy: -1`` entries in the ``_codegen`` YAMLs
are inputs -- so without this, a regression to the pool estimate would be
invisible and would quietly change kernel selection at runtime.

Two invariants:

1. **Occupancy matches the descriptor.** For every emitted kernel,
   ``kernel["CUOccupancy"]`` equals ``getOccupancy()`` of the counts the kernel
   actually declares.  This is the invariant; it is exact, not a bound.

2. **The corpus still exercises the correction.** At least one kernel must
   declare fewer registers than its pool reserved *and* reach a different
   occupancy because of it.  Without that the first invariant holds vacuously --
   both formulas agree whenever the counts agree -- and a revert to the pool
   estimate would pass unnoticed.

CPU-only: no GPU required.
"""

import os
import re

import pytest

import codegen_harness as _ch
import config_harness as _cfgh

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

# A config whose VGPR pool is far looser than what the emitted code needs, which
# is what makes invariant 2 bite. If it ever stops being so, the fixture says so
# rather than letting the suite pass on a vacuous comparison.
_CONFIG = os.path.join(
    os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250",
    "rich_gemm_sb.yaml",
)
_LIMIT = 2

_VGPR_RE = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")
_SGPR_RE = re.compile(r"\.amdhsa_next_free_sgpr\s+(\d+)")


def _emit_with_occupancy(config_path, arch, limit):
    """Emit kernels, snapshotting the declared counts against the pool estimate.

    ``processKernelSource`` leaves the writer holding the last kernel's state, so
    everything derived from ``kwa.states`` is read inside the loop.
    """
    import rocisa  # noqa: F401
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    assembler, isaInfoMap = _cfgh._toolchain_for(arch)
    results = []
    with _cfgh._isolated_globals_with_isa(isaInfoMap):
        solutions = _cfgh._solutions_from_config_unguarded(
            config_path, assembler, isaInfoMap, limit_solutions=limit
        )
        kernels = generateKernelObjectsFromSolutions(solutions)
        kernels = sorted(kernels, key=lambda k: getKernelFileBase(False, k))[:limit]
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        for kernel in kernels:
            ri = _ch._init_rocisa_for(kernel)
            base = _ch._prepare_kernel(kernel, False)
            res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            src = src or ""

            declVgpr = _VGPR_RE.search(src)
            declSgpr = _SGPR_RE.search(src)
            assert declVgpr and declSgpr, f"{base}: emitted source declares no register counts"

            poolVgpr, poolSgpr = kwa.vgprPool.size(), kwa.sgprPool.size()
            agprs, lds = kwa.agprPool.size(), kwa.getLdsSize(kernel)

            def occupancy(vgprs, sgprs):
                return kwa.getOccupancy(kernel["NumThreads"], vgprs, sgprs, lds, agprs,
                                        kwa.states.doubleVgpr)

            results.append({
                "base": base,
                "err": res.err,
                "declVgpr": int(declVgpr.group(1)),
                "declSgpr": int(declSgpr.group(1)),
                "poolVgpr": poolVgpr,
                "poolSgpr": poolSgpr,
                "recordedOccupancy": kernel["CUOccupancy"],
                "declaredOccupancy": occupancy(int(declVgpr.group(1)), int(declSgpr.group(1))),
                "poolOccupancy": occupancy(poolVgpr, poolSgpr),
            })
    return results


@pytest.fixture(scope="module")
def emission(tmp_path_factory):
    """Emit once, with ENABLE_DEBUG_STINKYTOFU_ASM on.

    The switch only writes Tensile's and StinkyTofu's assembly side by side under
    ``cmpasm/`` in the working directory, so one emission serves every test here,
    including the one about those files.
    """
    workDir = tmp_path_factory.mktemp("declared_counts")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ENABLE_DEBUG_STINKYTOFU_ASM", "1")
        mp.chdir(workDir)
        results = _emit_with_occupancy(_CONFIG, _ARCH, _LIMIT)
    return results, workDir / "cmpasm"


@pytest.fixture(scope="module")
def emitted(emission):
    results, _ = emission
    assert results, "no kernels emitted; the config or the harness went stale"
    assert all(r["err"] == 0 for r in results), (
        "every kernel must emit cleanly, or the occupancy recorded for it means nothing: %s"
        % [(r["base"], r["err"]) for r in results if r["err"] != 0]
    )
    return results


def test_occupancy_follows_the_declared_counts(emitted):
    """Invariant 1: the recorded occupancy is the declared counts' occupancy."""
    wrong = [
        (r["base"], r["recordedOccupancy"], r["declaredOccupancy"],
         r["declVgpr"], r["declSgpr"])
        for r in emitted
        if r["recordedOccupancy"] != r["declaredOccupancy"]
    ]
    assert not wrong, (
        "CUOccupancy does not match the emitted descriptor for "
        "(kernel, recorded, expected, declared vgprs, declared sgprs): %s. "
        "The hardware reserves what .amdhsa_next_free_* declares, so this is the "
        "number that reaches sizeMapping.CUOccupancy and drives Stream-K grid "
        "sizing at runtime." % wrong
    )


def test_the_corpus_still_exercises_the_correction(emitted):
    """Invariant 2: at least one kernel where the pool estimate disagrees.

    Guards the other direction. If every kernel here declared what its pool
    reserved, the assertion above would be comparing a number with itself and a
    revert to the pool estimate would pass it.
    """
    compacted = [r for r in emitted if r["declVgpr"] < r["poolVgpr"]]
    assert compacted, (
        "no kernel declares fewer VGPRs than its pool reserved, so the emit-time "
        "refresh is doing nothing here and the invariant above is vacuous. Point "
        "_CONFIG at a config with a looser pool. Saw (declared, pool): %s"
        % [(r["declVgpr"], r["poolVgpr"]) for r in emitted]
    )

    moved = [r for r in emitted if r["poolOccupancy"] != r["declaredOccupancy"]]
    assert moved, (
        "every kernel reaches the same occupancy from its pool estimate as from "
        "its declared counts, so this file cannot tell the two apart and would "
        "pass against the pre-fix behaviour. Point _CONFIG at a config whose "
        "compaction crosses an occupancy granule. Saw (pool occ, declared occ, "
        "declared vgprs, pool vgprs): %s"
        % [(r["poolOccupancy"], r["declaredOccupancy"], r["declVgpr"], r["poolVgpr"])
           for r in emitted]
    )


def _declared_counts(path):
    text = path.read_text()
    return int(_VGPR_RE.search(text).group(1)), int(_SGPR_RE.search(text).group(1))


def test_the_debug_comparison_keeps_tensiles_own_counts(emitted, emission):
    """``orig/`` declares the counts Tensile reserved, ``st/`` the ones emitted.

    Re-judging the budget sets the kernel body's declared counts to the emitted
    ones, so a dump of Tensile's assembly taken after it declared the reallocated
    counts too, and the two files agreed on exactly what the comparison is for.
    Invariant 2 above guarantees a kernel whose counts differ, so this cannot
    pass by comparing equal numbers.
    """
    _, cmpasm = emission
    orig = sorted(_declared_counts(p) for p in (cmpasm / "orig").glob("*.s"))
    st = sorted(_declared_counts(p) for p in (cmpasm / "st").glob("*.s"))
    assert orig == sorted((r["poolVgpr"], r["poolSgpr"]) for r in emitted), (
        "cmpasm/orig does not declare Tensile's own counts: %s" % orig
    )
    assert st == sorted((r["declVgpr"], r["declSgpr"]) for r in emitted), (
        "cmpasm/st does not declare the emitted counts: %s" % st
    )


def test_recorded_occupancy_is_usable(emitted):
    """A non-positive occupancy reaches the runtime as max(occ, 1) and hides a bug."""
    bad = [(r["base"], r["recordedOccupancy"]) for r in emitted if r["recordedOccupancy"] < 1]
    assert not bad, f"CUOccupancy <= 0 after codegen for: {bad}"
