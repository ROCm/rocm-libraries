# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 StreamK + TDM + PAP + PrefetchGL2 SGPR-budget characterization (CPU-only).

The MX StreamK+PAP+TDM family on gfx1250 runs within a couple of registers of the
106-SGPR cap, and ``PrefetchGL2=1`` + ``StreamKForceDPOnly=0`` is its tightest
corner. When a kernel crosses the cap, ``KernelWriterAssembly.checkResources``
does not fail the build: it sets ``states.overflowedResources = 2`` and rewrites
the body to ``s_endpgm`` wrapped in ``.if 0``. The kernel then launches, writes
nothing, and only shows up as a numerical mismatch in a GPU test.

These tests therefore assert the budget directly on full-kernel codegen, so an
SGPR-lifetime change that pushes this family over the cap is caught on CPU.

The second test pins the invariant that the TDM stagger wave-parity path must
hold: ``sgpr("WaveIdx")`` may only be read while it is defined. The parity
emitters decide per kernel whether to read ``WaveIdx`` directly or recompute the
parity from ``vgpr("Serial")``, and ``setupNewTile`` decides whether to release
``WaveIdx`` early; if those two decisions ever disagree the emit references a
symbol that is ``UNDEF``. Nothing here assembles the output, so ``err == 0``
proves nothing about symbols -- the check has to be explicit.

CPU-only: no GPU required.
"""

import os
import re

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_tdm_prefetchgl2.yaml",
)

# gfx1250 wave32 architected SGPR cap (rocisa hardware_caps.hpp: MaxSgpr).
_MAX_SGPR_GFX1250 = 106


@pytest.fixture(scope="module")
def emitted():
    """``[(basename, src, err), ...]`` plus the per-kernel register accounting.

    ``checkResources`` runs once per kernel with the final pool sizes and is
    where the overflow verdict is reached, so wrapping it is the only way to see
    ``sgprPool.size()`` and ``states.overflowedResources`` from outside the emit.
    """
    from Tensile.KernelWriterAssembly import KernelWriterAssembly

    records = []
    original = KernelWriterAssembly.checkResources

    def recording(self, kernel, mkb):
        result = original(self, kernel, mkb)
        records.append(
            {
                "name": self.states.kernelName,
                "sgprs": self.sgprPool.size(),
                "maxSgpr": self.states.regCaps["MaxSgpr"],
                "overflowedResources": self.states.overflowedResources,
            }
        )
        return result

    KernelWriterAssembly.checkResources = recording
    try:
        results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH, canonical=False)
    finally:
        KernelWriterAssembly.checkResources = original

    decoded = [
        (base, src.decode(errors="replace") if isinstance(src, (bytes, bytearray)) else (src or ""), err)
        for base, src, err in results
    ]
    return decoded, records


def test_streamk_tdm_prefetchgl2_gfx1250_fits_sgpr_budget(emitted):
    """Every kernel of the PrefetchGL2 x StreamKForceDPOnly sweep fits the cap."""
    results, records = emitted
    assert len(results) == 4, f"Expected the 2x2 PGL2 x SKFDPO sweep, got {len(results)}"
    assert records, "checkResources never ran; the emit path changed"

    assert all(r["maxSgpr"] == _MAX_SGPR_GFX1250 for r in records), (
        f"Expected MaxSgpr {_MAX_SGPR_GFX1250} for {_ARCH}, got "
        f"{sorted({r['maxSgpr'] for r in records})}"
    )

    over = [r for r in records if r["sgprs"] > r["maxSgpr"]]
    assert not over, (
        "SGPR pool exceeded the cap: "
        + ", ".join(f"{r['name']}: {r['sgprs']} > {r['maxSgpr']}" for r in over)
    )

    flagged = [r for r in records if r["overflowedResources"] != 0]
    assert not flagged, (
        "checkResources flagged overflowedResources (kernel is replaced by a bare "
        "s_endpgm and silently writes nothing): "
        + ", ".join(f"{r['name']}: code {r['overflowedResources']}" for r in flagged)
    )

    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} failed to emit (err={err})"
        assert "overflowed resources" not in src, (
            f"Kernel {base!r} emitted the overflow stub instead of a kernel body"
        )
        nextFree = re.search(r"\.amdhsa_next_free_sgpr\s+(\d+)", src)
        assert nextFree, f"Kernel {base!r} has no .amdhsa_next_free_sgpr"
        assert int(nextFree.group(1)) <= _MAX_SGPR_GFX1250, (
            f"Kernel {base!r} declares {nextFree.group(1)} SGPRs, cap is {_MAX_SGPR_GFX1250}"
        )


def test_streamk_tdm_prefetchgl2_gfx1250_waveidx_not_read_after_undef(emitted):
    """No kernel reads sgprWaveIdx after the symbol is undefined."""
    results, _records = emitted

    checked = 0
    for base, src, _err in results:
        lines = src.splitlines()
        if not any(re.match(r"\s*\.set sgprWaveIdx, \d", line) for line in lines):
            # Kernel body was replaced by the overflow stub; that is the budget
            # test's business, and there is no liveness range to check here.
            continue

        # Only the wave-separated stagger path is in scope: it is the one where
        # the release point and the parity reads are decided independently.
        if not any("// check wave parity" in line or "// wave parity (A=even/B=odd)" in line
                   for line in lines):
            continue
        checked += 1

        undef = [i for i, line in enumerate(lines) if re.match(r"\s*\.set sgprWaveIdx, UNDEF", line)]
        assert len(undef) == 1, (
            f"Kernel {base!r} undefines sgprWaveIdx {len(undef)} times, expected exactly 1"
        )

        stale = [
            (i + 1, lines[i].strip())
            for i in range(undef[0] + 1, len(lines))
            if "sgprWaveIdx" in lines[i]
        ]
        assert not stale, (
            f"Kernel {base!r} reads sgprWaveIdx after .set sgprWaveIdx, UNDEF "
            f"(line {undef[0] + 1}): {stale[:5]}"
        )

    assert checked, "No kernel took the TDM stagger wave-parity path; config drifted"
