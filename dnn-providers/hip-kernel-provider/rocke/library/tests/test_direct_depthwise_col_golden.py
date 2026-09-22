# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the column-streamed depthwise conv kernel.

``build_direct_depthwise_col`` is covered by the Python/C++ byte-identity gate:
``tests/parity/conv_direct_grouped_emit.{py,c}`` config indices 12-19 pin the
two engines against each other across strides 1/2/3, all three dtypes and both
tile-guard elision paths. This test is the *other* half and does a different
job: byte-identity says the two engines agree, while these hashes say the
emission has not moved at all. A change that edits both engines in lockstep
keeps the gate GREEN and still lands here, which is what makes a refactor that
is meant to be behaviour-preserving *shown* to be, rather than argued to be.

The stride-1 fp16 cases carry the extra weight. The stride generalization
(``n_iters``/tap-pruning/accumulator-index formulas parameterized by
``p.stride``) claims to reduce to the original expressions at ``stride == 1``;
these hashes are what turns that claim into a check.

Run or re-bless from ``rocke/library``::

    python -m pytest tests/test_direct_depthwise_col_golden.py
    python tests/test_direct_depthwise_col_golden.py --write

Review the IR change before re-blessing the hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

_TESTS = Path(__file__).resolve().parent
_LIBRARY = _TESTS.parent
_PLATFORM_PYTHON = _LIBRARY.parent / "platform" / "python"
_GOLDEN = _TESTS / "golden" / "direct_depthwise_col_ir_sha256.json"
_GOLDEN_GFX942 = _TESTS / "golden" / "direct_depthwise_col_gfx942_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22", "llvm23")
_ARCH = "gfx950"
_SCHEMA = "direct_depthwise_col.ir_golden_sha256/v1"

# The library path must precede tests/ so tests/dispatch cannot shadow the real
# dispatch package when this file is executed directly.
for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cases() -> dict[str, Callable]:
    """Return builders spanning the column-streamed depthwise knob space.

    Each entry pins one axis the emitter branches on: the static tap grid
    (``KH``, ``PAD``, ``stride``), the two ``*_tile_exact`` guard elisions, the
    accumulator band width (``block_w``), the channel tiling (``block_waves``),
    and the element type.
    """
    from kernels.common.conv_direct_grouped import (
        DirectConvProblem,
        DirectDepthwiseColSpec,
        build_direct_depthwise_col,
    )

    cases: dict[str, Callable] = {}
    kernel_names: dict[str, str] = {}

    def add(cid, *, block_w=1, block_waves=1, dtype="fp16", **pkw):
        # ``stride`` rides through **pkw onto DirectConvProblem; ``dtype`` lives
        # on the spec, since DirectConvProblem is shared with the builders that
        # are still fp16-only.
        problem = DirectConvProblem(cpg=1, kpg=1, **pkw)
        spec = DirectDepthwiseColSpec(
            problem=problem,
            name=f"golden_dwcol_{cid}",
            block_w=block_w,
            block_waves=block_waves,
            dtype=dtype,
        )
        name = spec.kernel_name()
        assert name not in kernel_names, (
            f"depthwise-col kernel-name collision: {cid!r} and "
            f"{kernel_names[name]!r} both use {name!r}"
        )
        kernel_names[name] = cid
        cases[cid] = lambda spec=spec: build_direct_depthwise_col(spec, arch=_ARCH)

    # --- stride 1, fp16: the emission Part 1 must leave untouched -----------
    # 3x3 same-padding, the common depthwise stage. bw1 and bw4 pin both the
    # scalar and the widest SLP-friendly accumulator band.
    add("s1_k3_h28_bw1", N=1, H=28, W=28, groups=64, KH=3, KW=3, PAD=1)
    add("s1_k3_h28_bw4", N=1, H=28, W=28, groups=64, KH=3, KW=3, PAD=1, block_w=4)
    # 7x7, and a second wave per block (block_ch 128).
    add("s1_k7_h14_bw2", N=2, H=14, W=14, groups=64, KH=7, KW=7, PAD=3, block_w=2)
    add(
        "s1_k7_h14_2wv",
        N=1, H=14, W=14, groups=128, KH=7, KW=7, PAD=3, block_waves=2,
    )
    # Large filter: the regime where the preload variant cannot be built and
    # this kernel is the only option. live_f32 = 14 + 31.
    add("s1_k31_h14_bw1", N=1, H=14, W=14, groups=64, KH=31, KW=31, PAD=15)
    # KH != KW -- the KW-independence that motivates the whole design.
    add("s1_k3x31_h16_bw1", N=1, H=16, W=32, groups=64, KH=3, KW=31, PAD=1)
    # Valid padding (PAD=0) and the degenerate 1x1 filter.
    add("s1_k3_pad0", N=1, H=10, W=10, groups=64, KH=3, KW=3, PAD=0)
    add("s1_k1_pad0", N=1, H=12, W=12, groups=64, KH=1, KW=1, PAD=0)
    # Tail guards: Wo % block_w != 0 keeps the store predicate, and
    # groups % block_ch != 0 keeps the channel predicate. Both are elided in
    # the cases above, so these pin the un-elided emission.
    add("s1_wtail_bw2", N=1, H=15, W=15, groups=64, KH=3, KW=3, PAD=1, block_w=2)
    add("s1_chtail", N=1, H=8, W=8, groups=100, KH=3, KW=3, PAD=1)

    # --- stride > 1: blessed as new entries, IR reviewed by hand -------------
    # At stride S the tap list for each unrolled row y keeps only the r with
    # (y - r) % S == 0, so the emitted grid is periodically ragged rather than
    # rectangular. These pin that pruning: a formula that drops or duplicates a
    # row changes the hash even though the kernel would still run.
    add("s2_k3_h28_bw2", N=1, H=28, W=28, groups=64, KH=3, KW=3, PAD=1,
        stride=2, block_w=2)
    add("s3_k3_h28_bw1", N=1, H=28, W=28, groups=64, KH=3, KW=3, PAD=1, stride=3)
    # Ragged on both axes at once: 5x5 at stride 3, and Wo=10 % block_w=3 != 0
    # so the W tail guard is live too.
    add("s3_k5_h28_bw3", N=1, H=28, W=28, groups=64, KH=5, KW=5, PAD=2,
        stride=3, block_w=3)
    # PAD=2 > (KH-1)/2=1: the padded input overhangs the input, which is what
    # `n_iters = (Ho-1)*stride + KH` exists to cover. Unreachable at stride 1 --
    # there the same overhang makes Ho > H and the validator rejects it.
    add("s2_pad2_overhang", N=1, H=12, W=12, groups=64, KH=3, KW=3, PAD=2, stride=2)
    # Large filter at stride > 1: live_f32 = Ho*1 + KH = 16 + 31.
    add("s2_k31_h32", N=1, H=32, W=32, groups=64, KH=31, KW=31, PAD=15, stride=2)

    # --- bf16 / fp32 ---------------------------------------------------------
    # Same geometry as s1_k3_h28_bw1 at the other two element types, so the diff
    # between these three hashes is exactly the load/convert/store width.
    add("bf16_s1_k3_h28", N=1, H=28, W=28, groups=64, KH=3, KW=3, PAD=1,
        dtype="bf16")
    add("fp32_s1_k3_h28", N=1, H=28, W=28, groups=64, KH=3, KW=3, PAD=1,
        dtype="fp32")
    # Element type crossed with the strided tap pruning and with a live W tail:
    # the two are independent code paths and both have to be right at once.
    add("bf16_s2_k5_bw3", N=1, H=28, W=28, groups=64, KH=5, KW=5, PAD=2,
        stride=2, block_w=3, dtype="bf16")
    add("fp32_s3_chtail", N=1, H=28, W=28, groups=100, KH=3, KW=3, PAD=1,
        stride=3, block_w=4, dtype="fp32")

    return cases


def _current_flavor() -> str:
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build: Callable, arch: str, flavor: str) -> tuple[str, int]:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=arch, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc(cases: dict, arch: str) -> dict:
    return {
        "schema": _SCHEMA,
        "flavors": {
            flavor: {
                "cases": {
                    cid: {"sha256": sha, "bytes": nbytes}
                    for cid, build in cases.items()
                    for sha, nbytes in [_sha_for(build, arch, flavor)]
                }
            }
            for flavor in _FLAVORS
        },
    }


def _cases_gfx942() -> dict[str, Callable]:
    """One representative case per key emission branch, targeting gfx942.

    Covers the arch-specific VGPR budget path (512 VGPRs on gfx942) and the
    gfx942 codegen target, which the gfx950-only golden cases do not exercise.
    """
    from kernels.common.conv_direct_grouped import (
        DirectConvProblem,
        DirectDepthwiseColSpec,
        build_direct_depthwise_col,
    )

    _arch = "gfx942"
    cases: dict[str, Callable] = {}

    def add(cid, *, block_w=4, block_waves=2, dtype="fp16", **pkw):
        problem = DirectConvProblem(cpg=1, kpg=1, **pkw)
        spec = DirectDepthwiseColSpec(
            problem=problem,
            name=f"golden_dwcol_942_{cid}",
            block_w=block_w,
            block_waves=block_waves,
            dtype=dtype,
        )
        cases[cid] = lambda spec=spec: build_direct_depthwise_col(spec, arch=_arch)

    # stride=1 fp16 — exercises the gfx942 VGPR budget + codegen target
    add("s1_k3_h8_bw4", N=2, H=8, W=8, groups=128, KH=3, KW=3, PAD=1)
    # stride=2 bf16 with both tile guards live — exercises strided tap pruning on gfx942
    add("s2_bf16_tail", N=1, H=9, W=9, groups=70, KH=3, KW=3, PAD=1, stride=2,
        block_w=4, block_waves=1, dtype="bf16")

    return cases


def _check_golden(golden_path: Path, cases: dict, arch: str) -> None:
    golden = json.loads(golden_path.read_text())
    assert golden.get("schema") == _SCHEMA

    flavor = _current_flavor()
    assert flavor in golden.get("flavors", {}), (
        f"no depthwise-col golden recorded for LLVM flavor {flavor!r} "
        f"(arch={arch}); review and re-bless the fixture"
    )

    recorded = golden["flavors"][flavor]["cases"]
    assert set(recorded) == set(cases), (
        f"depthwise-col golden case set drifted (arch={arch}): "
        f"recorded={sorted(recorded)}, current={sorted(cases)}"
    )

    drift = []
    for cid, build in cases.items():
        want = recorded[cid]["sha256"]
        got, nbytes = _sha_for(build, arch, flavor)
        if got != want:
            drift.append(
                f"{cid}: {want} -> {got} "
                f"({recorded[cid]['bytes']} -> {nbytes} bytes)"
            )
    assert not drift, (
        f"depthwise-col LLVM IR drift vs golden (arch={arch}):\n  "
        + "\n  ".join(drift)
    )


def test_direct_depthwise_col_ir_matches_golden():
    assert _GOLDEN.exists(), (
        "missing depthwise-col golden fixture; generate it with "
        f"`python {Path(__file__).name} --write`"
    )
    _check_golden(_GOLDEN, _cases(), _ARCH)


def test_direct_depthwise_col_ir_matches_golden_gfx942():
    if not _GOLDEN_GFX942.exists():
        import pytest

        pytest.skip(
            "gfx942 depthwise-col golden not yet blessed; "
            f"generate with `python {Path(__file__).name} --write-942`"
        )
    _check_golden(_GOLDEN_GFX942, _cases_gfx942(), "gfx942")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(
            json.dumps(_build_doc(_cases(), _ARCH), indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {_GOLDEN}")
    elif "--write-942" in sys.argv:
        _GOLDEN_GFX942.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN_GFX942.write_text(
            json.dumps(_build_doc(_cases_gfx942(), "gfx942"), indent=2, sort_keys=True)
            + "\n"
        )
        print(f"wrote {_GOLDEN_GFX942}")
    else:
        test_direct_depthwise_col_ir_matches_golden()
