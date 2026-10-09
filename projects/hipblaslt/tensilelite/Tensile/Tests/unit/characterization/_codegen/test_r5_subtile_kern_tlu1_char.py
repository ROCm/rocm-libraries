# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""R5 — gfx950 bf16 NN TLU=1 subtile codegen characterization.

Drives the designed BenchmarkProblems config
(``data/test_data/_designed/gfx950/subtile_kern_tlu1.yaml``) through the
config-driven emit harness.  That config was listed in ``input_yaml_files.txt``
but no test module referenced it, so nothing ran it; this module is its driver.

It is the codegen counterpart to two tests that stop short of full emit:

  * ``test_r5_subtile_kern_char.py`` constructs ``TileInfo`` directly with a
    TLU1 geometry and checks the ``contiguousDim == 'M'`` branches.  It never
    reaches an emitter.
  * ``Tests/common/gemm/gfx950/subtile_bf16_tlu1.yaml`` validates the same
    geometry numerically, but needs a GPU and the client.

Here the kernels are emitted on CPU and the assembly text itself is asserted
on, which is the only one of the three that can say *which arm ran*.

CPU-only. No GPU, no compile, no hardware access.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "subtile_kern_tlu1.yaml",
)

# MacroTile0 = MatrixInstM * MIWaveTile[0] * MIWaveGroup[0], and a bf16 strip
# holds 4 MMA-M tiles (64 rows).  Of the three MatrixInstruction entries in the
# config only MT64x64 leaves MacroTile0 / MIWaveGroup[0] = 32 < 64, so it is the
# one where two waves share a strip.
_SHARED_STRIP_MT = "MT64x64x64"


def test_r5_subtile_kern_tlu1_gfx950_emits_assembly():
    """The NN TLU=1 config emits real gfx950 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) == 3, (
        f"config has 3 MatrixInstruction forks, got {len(results)} kernels"
    )
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"kernel {base!r}: suspiciously short assembly"
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"


def test_r5_subtile_kern_tlu1_gfx950_takes_the_tlu1_arm():
    """Every kernel goes down the TLU=1 arm, and only the shared-strip tile folds a window.

    ``err == 0`` alone would also hold if A had silently fallen back to the
    TLU=0 row-major geometry, so the three things that distinguish the arm are
    asserted directly:

    1. ``A TLU1:`` address-arithmetic comments — the M-contiguous GR/LR offset
       chain, absent from a TLU=0 kernel.
    2. ``ds_read_b64_tr_b16`` — the transposed LDS read the stacked geometry
       exists to feed.
    3. The wave window folded in *before* the XOR on the shared-strip tile.
       Adding it after gives ``(tile_m ^ swz) + window`` and corrupts every wave
       with a non-zero window; the emitted comment records which side it landed
       on, and the two unshared tiles must not carry it at all.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)

    for base, src, _err in results:
        assert "A TLU1:" in src, (
            f"kernel {base!r}: no TLU1 address arithmetic -- A fell back to the "
            f"K-contiguous geometry"
        )
        assert "ds_read_b64_tr_b16" in src, (
            f"kernel {base!r}: no transposed LDS read"
        )

    shared = [(b, s) for (b, s, _e) in results if _SHARED_STRIP_MT in b]
    assert len(shared) == 1, f"expected exactly one {_SHARED_STRIP_MT} kernel, got {len(shared)}"
    base, src = shared[0]
    assert "wave window (pre-swizzle" in src, (
        f"kernel {base!r}: waves share a strip here, so the window must be folded "
        f"in before the XOR"
    )

    for b, s, _e in results:
        if _SHARED_STRIP_MT not in b:
            assert "wave window" not in s, (
                f"kernel {b!r}: one wave per strip, so there is no window to fold"
            )


def test_r5_subtile_kern_tlu1_gfx950_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for every emitted TLU=1 kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
