# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR stability test for the gfx950 KDA kernel family.

This is the per-family deep lane for chunkwise KDA. It pins both chunk sizes,
both scan atom widths, the split and fused paths, state ABI flags, and every
supported value-split geometry. Raw-input fusion and the fused LDS overlay are
also represented. The test only lowers Python to LLVM IR; no GPU or comgr is
required.

Run or re-bless from ``rocke/library``::

    python tests/run_all.py --only kda_gfx950
    python tests/test_kda_gfx950_golden.py --write

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
_GOLDEN = _TESTS / "golden" / "kda_gfx950_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22", "llvm23")
_ARCH = "gfx950"

# The library path must precede tests/ so tests/dispatch cannot shadow the real
# dispatch package when this file is executed directly.
for _path in (str(_LIBRARY), str(_PLATFORM_PYTHON)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _cases() -> dict[str, Callable]:
    """Return builders spanning the gfx950 KDA code-generation knob space."""
    from kernels.gfx950.kda_chunkwise import (
        KdaChunkFusedSpec,
        KdaChunkPrepSpec,
        KdaChunkScanSpec,
        KdaTileSpec,
        build_kda_chunk_fused,
        build_kda_chunk_prep,
        build_kda_chunk_scan,
        tuned_kda_chunk_scan_spec,
    )

    cases: dict[str, Callable] = {}
    kernel_names: dict[str, str] = {}

    def add(cid, spec, builder):
        name = spec.kernel_name()
        assert name not in kernel_names, (
            f"KDA kernel-name collision: {cid!r} and {kernel_names[name]!r} "
            f"both use {name!r}"
        )
        kernel_names[name] = cid
        cases[cid] = lambda spec=spec, builder=builder: builder(spec, arch=_ARCH)

    # C16 needs an explicit zero-padded atom schedule. Reuse it in both halves
    # of the split path and in the fused path so all three emissions are pinned.
    c16 = KdaTileSpec(
        chunk=16,
        block_size=512,
        pad_cb=16,
        tile_atom_m=16,
        scan_atom_m=16,
    )

    add("kda_gfx950/split_c32_prep", KdaChunkPrepSpec(), build_kda_chunk_prep)
    add(
        "kda_gfx950/split_c16_prep",
        KdaChunkPrepSpec(tile=c16),
        build_kda_chunk_prep,
    )
    add(
        "kda_gfx950/split_c32_prep_raw_fusions",
        KdaChunkPrepSpec(
            raw_inputs=True,
            fuse_qk_l2norm=True,
            fuse_gate=True,
            fuse_beta_sigmoid=True,
            has_dt_bias=True,
        ),
        build_kda_chunk_prep,
    )
    add(
        "kda_gfx950/split_c32_prep_raw_fusions_beta_fp32",
        KdaChunkPrepSpec(
            raw_inputs=True,
            fp32_beta_dtype=True,
            fuse_qk_l2norm=True,
            fuse_gate=True,
            fuse_beta_sigmoid=True,
            has_dt_bias=True,
        ),
        build_kda_chunk_prep,
    )

    # The default standalone scan is C32/SA32/value_splits=1.
    add("kda_gfx950/split_c32_scan_sa32", KdaChunkScanSpec(), build_kda_chunk_scan)
    add(
        "kda_gfx950/split_c32_scan_no_prefetch",
        KdaChunkScanSpec(prefetch_tiles=False),
        build_kda_chunk_scan,
    )
    add(
        "kda_gfx950/split_c32_scan_tuned_vs4",
        tuned_kda_chunk_scan_spec(96),
        build_kda_chunk_scan,
    )
    add(
        "kda_gfx950/split_c16_scan_sa16",
        KdaChunkScanSpec(tile=c16),
        build_kda_chunk_scan,
    )
    add(
        "kda_gfx950/split_c32_scan_h0",
        KdaChunkScanSpec(has_initial_state=True),
        build_kda_chunk_scan,
    )
    add(
        "kda_gfx950/split_c32_scan_noht",
        KdaChunkScanSpec(store_final_state=False),
        build_kda_chunk_scan,
    )
    add(
        "kda_gfx950/split_c32_scan_h0_noht",
        KdaChunkScanSpec(has_initial_state=True, store_final_state=False),
        build_kda_chunk_scan,
    )
    for value_splits, block_size, scan_atom_m in (
        (2, 128, 0),
        (4, 64, 0),
        (8, 64, 16),
    ):
        add(
            f"kda_gfx950/split_c32_scan_vs{value_splits}",
            KdaChunkScanSpec(
                tile=KdaTileSpec(
                    block_size=block_size,
                    scan_atom_m=scan_atom_m,
                ),
                value_splits=value_splits,
                token_major_io=True,
            ),
            build_kda_chunk_scan,
        )

    # The tuned fused default is C32/SA16; an untuned C32 tile pins SA32.
    add("kda_gfx950/fused_c32_sa16", KdaChunkFusedSpec(), build_kda_chunk_fused)
    add(
        "kda_gfx950/fused_c32_sa32",
        KdaChunkFusedSpec(tile=KdaTileSpec()),
        build_kda_chunk_fused,
    )
    add(
        "kda_gfx950/fused_c16_sa16",
        KdaChunkFusedSpec(tile=c16),
        build_kda_chunk_fused,
    )
    add(
        "kda_gfx950/fused_c32_h0",
        KdaChunkFusedSpec(has_initial_state=True),
        build_kda_chunk_fused,
    )
    add(
        "kda_gfx950/fused_c32_noht",
        KdaChunkFusedSpec(store_final_state=False),
        build_kda_chunk_fused,
    )
    add(
        "kda_gfx950/fused_c32_h0_noht",
        KdaChunkFusedSpec(has_initial_state=True, store_final_state=False),
        build_kda_chunk_fused,
    )
    add(
        "kda_gfx950/fused_c32_overlay",
        KdaChunkFusedSpec(
            tile=KdaTileSpec(),
            prefetch_inputs=False,
            overlay_lds=True,
        ),
        build_kda_chunk_fused,
    )

    return cases


def _current_flavor() -> str:
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build: Callable, flavor: str) -> tuple[str, int]:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc() -> dict:
    cases = _cases()
    return {
        "schema": "kda_gfx950.ir_golden_sha256/v1",
        "flavors": {
            flavor: {
                "cases": {
                    cid: {"sha256": sha, "bytes": nbytes}
                    for cid, build in cases.items()
                    for sha, nbytes in [_sha_for(build, flavor)]
                }
            }
            for flavor in _FLAVORS
        },
    }


def test_kda_gfx950_ir_matches_golden():
    assert _GOLDEN.exists(), (
        "missing gfx950 KDA golden fixture; generate it with "
        f"`python {Path(__file__).name} --write`"
    )
    golden = json.loads(_GOLDEN.read_text())
    assert golden.get("schema") == "kda_gfx950.ir_golden_sha256/v1"

    flavor = _current_flavor()
    assert flavor in golden.get("flavors", {}), (
        f"no gfx950 KDA golden recorded for LLVM flavor {flavor!r}; "
        "review and re-bless the fixture"
    )

    cases = _cases()
    recorded = golden["flavors"][flavor]["cases"]
    assert set(recorded) == set(cases), (
        "gfx950 KDA golden case set drifted: "
        f"recorded={sorted(recorded)}, current={sorted(cases)}"
    )

    drift = []
    for cid, build in cases.items():
        want = recorded[cid]["sha256"]
        got, nbytes = _sha_for(build, flavor)
        if got != want:
            drift.append(
                f"{cid}: {want} -> {got} ({recorded[cid]['bytes']} -> {nbytes} bytes)"
            )
    assert not drift, "gfx950 KDA LLVM IR drift vs golden:\n  " + "\n  ".join(drift)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_kda_gfx950_ir_matches_golden()
