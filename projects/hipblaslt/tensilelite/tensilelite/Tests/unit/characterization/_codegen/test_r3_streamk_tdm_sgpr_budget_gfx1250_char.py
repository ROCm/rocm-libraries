# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 StreamK + wave-separated TDM SGPR-budget regression coverage (CPU-only).

Two invariants, both asserted on fully generated kernels rather than on mocked
sub-module names:

1. **SGPR budget.** ``sgprPool.size()`` must stay within ``regCaps["MaxSgpr"]``
   (106 on gfx1250) and ``states.overflowedResources`` must be 0. When the pool
   overflows, ``checkResources`` does not fail the build -- it rewrites the kernel
   body into ``s_endpgm // overflowed resources`` wrapped in ``.if 0``, so the
   kernel launches, writes nothing, and only shows up as a numerical mismatch in
   a GPU test. This is the invariant that a held-too-long ``sgprWaveIdx`` broke:
   holding that low-indexed SGPR across the unroll loop cost the PrefetchGL2=1 /
   StreamKForceDPOnly=0 variant its last two free slots (105 -> 107).

2. **WaveIdx liveness.** No ``s[sgprWaveIdx]`` reference may appear after
   ``.set sgprWaveIdx, UNDEF``. The emitter runs no assembler, so ``err == 0``
   says nothing about symbol liveness -- an unaccounted-for parity consumer
   emitting a symbolic read past the undefine has to be caught explicitly here.

Scoped to the wave-separated stagger path (``enableTDMA and enableTDMB and
NumWaves > 1``), since the plain TDM path legitimately reads WaveIdx once and
then undefines it in the same prologue.

Run for MX-FP8 and MX-FP4 inputs, matching the two GPU tests that regressed. Both
currently land on the same pool sizes (93/97/101/105 of 106); FP4 is covered because
it reaches them through a different allocation path -- different global-read widths,
LDS layout and scale-block bookkeeping -- that could drift away from FP8.

CPU-only: no GPU required.
"""

import itertools
import os
import re

import pytest

import codegen_harness as _ch
import config_harness as _cfgh

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250"
)

# The fork axes both designed configs sweep, in variant-key order. This is the single
# source of truth: the expected variant set and the emit limit are both derived from
# it, so the count can never agree while the set silently drifts.
#
# Why pinning the set matters: the limit handed to the harness is also a truncation
# (config_harness slices the permutation list), and constructLazyForkPermutations
# walks the fork list in reverse, so the first-listed fork parameter varies fastest.
# Widening any fork list ahead of PrefetchGL2 in the YAML could fill the first N
# permutations with PrefetchGL2=0 only -- the count assertion would still pass while
# the tight corner went untested.
_FORK_AXES = (
    ("PGL", "PrefetchGL2", (0, 1)),
    ("SKFDPO", "StreamKForceDPOnly", (0, 1)),
    ("PGR", "PrefetchGlobalRead", (1, 2)),
    ("SIA", "ScheduleIterAlg", (0, 4)),
)


def _variant_key(lookup):
    """Render a variant label from a per-axis value lookup."""
    return "/".join(f"{label}{lookup(param)}" for label, param, _ in _FORK_AXES)


_EXPECTED_VARIANTS = frozenset(
    "/".join(f"{label}{value}" for (label, _, _), value in zip(_FORK_AXES, combo))
    for combo in itertools.product(*(values for _, _, values in _FORK_AXES))
)

# MUST equal the true fork product, or the harness truncates and the missing
# permutations are never emitted.
_EXPECTED_KERNELS = len(_EXPECTED_VARIANTS)

# The tightest variant: 105 of 106 SGPRs once WaveIdx is released before the loop,
# 107 while it is held. If this one is ever missing from the sweep, the budget
# assertion is running with spare headroom and proves nothing.
_TIGHT_VARIANT = "PGL1/SKFDPO0/PGR1/SIA0"

assert _TIGHT_VARIANT in _EXPECTED_VARIANTS, (
    f"_TIGHT_VARIANT {_TIGHT_VARIANT!r} is not in the _FORK_AXES product -- it went "
    "stale when the axes changed. Fix it to name the tightest surviving variant "
    "rather than deleting this check."
)

_CONFIGS = {
    "mxf8": os.path.join(_DATA_DIR, "streamk_tdm_prefetchgl2.yaml"),
    "mxf4": os.path.join(_DATA_DIR, "streamk_tdm_prefetchgl2_f4.yaml"),
}

_UNDEF_RE = re.compile(r"^\s*\.set\s+sgprWaveIdx\s*,\s*UNDEF\s*$", re.MULTILINE)
_WAVEIDX_READ_RE = re.compile(r"s\[sgprWaveIdx\]")


def _emit_with_reg_state(config_path, arch, limit):
    """Emit kernels, returning per-kernel assembly *and* register-pool state.

    ``config_harness.emit_kernels_from_config`` returns only ``(base, src, err)``;
    the SGPR budget assertion needs ``sgprPool.size()`` /
    ``states.overflowedResources`` / ``regCaps["MaxSgpr"]``, which live on the
    writer. Drive the same emit path and snapshot them per kernel.
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
        if not _ch._WARMED and kernels:
            _cfgh._emit_one(kwa, kernels[0], False, True)
            _ch._WARMED = True
        for kernel in kernels:
            ri = _ch._init_rocisa_for(kernel)
            base = _ch._prepare_kernel(kernel, False)
            res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            results.append(
                {
                    "base": base,
                    "src": src or "",
                    "variant": _variant_key(lambda param: kernel[param]),
                    "waveSeparated": bool(
                        kernel["enableTDMA"] and kernel["enableTDMB"] and kernel["NumWaves"] > 1
                    ),
                    "poolSize": kwa.sgprPool.size(),
                    "maxSgpr": kwa.states.regCaps["MaxSgpr"],
                    "overflowed": kwa.states.overflowedResources,
                }
            )
    return results


@pytest.fixture(scope="module", params=sorted(_CONFIGS))
def emitted(request):
    results = _emit_with_reg_state(_CONFIGS[request.param], _ARCH, _EXPECTED_KERNELS)
    dtype = request.param

    got = {r["variant"] for r in results}
    missing = _EXPECTED_VARIANTS - got
    unexpected = got - _EXPECTED_VARIANTS
    assert not missing and not unexpected, (
        f"{dtype}: the emitted sweep is not the expected fork product. "
        f"Missing variants: {sorted(missing)}. Unexpected variants: {sorted(unexpected)}. "
        f"{_TIGHT_VARIANT} is the tight corner -- if it is in the missing list, every "
        "assertion below is running on kernels with spare SGPR headroom and proves "
        "nothing. The emit limit is also a truncation, so widening a fork list in the "
        "YAML without updating _FORK_AXES silently drops permutations."
    )
    assert _TIGHT_VARIANT in got, f"{dtype}: tight corner {_TIGHT_VARIANT} not emitted"
    assert len(results) == _EXPECTED_KERNELS, (
        f"{dtype}: expected {_EXPECTED_KERNELS} kernels from the fork product, "
        f"got {len(results)} (duplicate variants?)"
    )
    assert all(r["waveSeparated"] for r in results), (
        f"{dtype}: config no longer selects the wave-separated TDM path: "
        f"{[(r['variant'], r['waveSeparated']) for r in results]}"
    )
    return results


def test_streamk_tdm_prefetchgl2_stays_within_sgpr_budget(emitted):
    """No variant in the fork product may exceed the gfx1250 SGPR cap."""
    over = [
        (r["variant"], r["poolSize"], r["maxSgpr"])
        for r in emitted
        if r["poolSize"] > r["maxSgpr"]
    ]
    assert not over, (
        "SGPR pool exceeded MaxSgpr for (variant, poolSize, maxSgpr): %s. "
        "checkResources replaces such kernels with s_endpgm, so they silently "
        "write nothing at runtime." % over
    )

    flagged = [(r["variant"], r["overflowed"]) for r in emitted if r["overflowed"] != 0]
    assert not flagged, f"states.overflowedResources != 0 for: {flagged}"


def test_streamk_tdm_prefetchgl2_emits_real_kernel_bodies(emitted):
    """An overflowed kernel is body-less; assert every variant kept its body."""
    for r in emitted:
        src, variant = r["src"], r["variant"]
        assert "overflowed resources" not in src, (
            f"{variant}: kernel body replaced by the resource-overflow stub"
        )
        assert re.search(r"^\s*\.if\s+0\s*$", src, re.MULTILINE) is None, (
            f"{variant}: kernel body wrapped in '.if 0' by checkResources"
        )
        m = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", src)
        assert m is not None, f"{variant}: no .amdhsa_next_free_sgpr in emitted source"
        assert int(m.group(1)) <= r["maxSgpr"], (
            f"{variant}: .amdhsa_next_free_sgpr={m.group(1)} exceeds "
            f"MaxSgpr={r['maxSgpr']}"
        )


def test_streamk_tdm_no_waveidx_read_after_undefine(emitted):
    """Every wave-parity consumer past the release must recompute from vgpr("Serial")."""
    for r in emitted:
        src, variant = r["src"], r["variant"]
        undef = _UNDEF_RE.search(src)
        assert undef is not None, (
            f"{variant}: sgprWaveIdx is never undefined, so it stays checked out "
            "across the whole kernel"
        )
        trailing = src[undef.end():]
        leaked = _WAVEIDX_READ_RE.findall(trailing)
        assert not leaked, (
            f"{variant}: {len(leaked)} s[sgprWaveIdx] reference(s) emitted after "
            ".set sgprWaveIdx, UNDEF -- a parity consumer is missing its "
            "isTdmWaveIdxLive fallback"
        )


def test_streamk_tdm_stagger_prologue_keeps_cheap_parity_read(emitted):
    """The stagger prologue must still read parity from s[sgprWaveIdx] directly.

    Guards the other direction: releasing WaveIdx before calculateStagger would
    silently downgrade the prologue to the vgpr("Serial") recompute.
    """
    for r in emitted:
        src, variant = r["src"], r["variant"]
        undef = _UNDEF_RE.search(src)
        assert undef is not None, f"{variant}: sgprWaveIdx is never undefined"
        prologue = src[: undef.end()]
        assert "s_bitcmp1_b32 s[sgprWaveIdx]" in prologue, (
            f"{variant}: no direct s[sgprWaveIdx] parity read before the release; "
            "the stagger prologue lost its fast path"
        )
