# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the gfx942 dense flash-attn kernel.

Hashes the Python-lowered LLVM IR (SHA256) of representative gfx942 ``attention_dense``
specs and compares against a checked-in per-flavor golden fixture, catching any
unintended codegen drift across the P1-P4 lever set. Pure text lowering -- no GPU / no
comgr required.

Covers the acceptance matrix: D64/D128 x bf16/fp16 x default/persistent x GQA, causal
and full. Every case is in the gfx942 supported set (varlen / ragged / sliding-window
are rejected on gfx942, so -- unlike the gfx950 sibling -- they are absent here).

Both D64 K-LDS layouts are pinned so drift on either is caught:
  * ``default_d64_*``  -- specs built DIRECTLY with ``lds_k_group_pad=0``: the UNPADDED
    layout, i.e. the A/B baseline the pad's ~2x is measured against.
  * ``dispatch_d64_*`` -- specs built through the gfx942 dispatch factory
    (``_dense_spec``), which inherits the shared default pad -> this GUARDS the padded
    IR the shipped D64 path actually emits. Building via the dispatch spec (rather than
    hard-coding the pad) means these cases auto-track any future D64 tuning change, so
    re-blessing stays a one-command operation across the kernel's evolution.

There is NO cpp/python byte-identity companion test: ``library/kernels/`` has no C++
engine mirror (settled when the port was scoped), so the dual-engine parity gate does
not apply to this kernel. The Python lowering IS the ground truth here.

COLLECTION: this is a CPU lane (pure text lowering), collected by the library
runner ``library/tests/run_all.py`` (default, non-``gpu`` lane) -- NOT by
``platform/tests/run_all.py``, which cannot reach the library tree (the one-way
library -> platform dependency rule). The actual CI pipeline invoking that runner
is a CI-team registration step, tracked as a follow-up. Run it and
re-bless manually:

    cd rocke/library
    python tests/run_all.py --only dense_gfx942          # runs this + the build lane
    PYTHONPATH=../platform/python:. python tests/test_attention_dense_gfx942_golden.py --write

Re-bless ONLY the gfx942 fixture; this test never reads or writes the gfx950 golden.
"""

import hashlib
import json
import sys
from pathlib import Path

_GOLDEN = (
    Path(__file__).resolve().parent / "golden" / "attention_dense_gfx942_ir_sha256.json"
)
_FLAVORS = ("llvm20", "llvm22")
_ARCH = "gfx942"

# Pin the ``library/`` root ahead of everything on sys.path. When this file is run
# directly (``python tests/...golden.py --write``) Python puts ``tests/`` on sys.path[0],
# where ``tests/dispatch/`` would otherwise shadow the real ``library/dispatch`` package
# that the dispatch-built (kpad-ON) golden cases import. Harmless under pytest.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


def _cases():
    """cid -> zero-arg builder returning a KernelDef. Small Sq keeps the IR compact
    while still exercising the full pipeline (both grid variants, both decodes).

    Two families of builders:
      * ``mk`` builds an ``AttentionDenseSpec`` DIRECTLY. The D64 cases pass
        ``lds_k_group_pad=0`` explicitly, pinning the UNPADDED K layout -- keep them:
        that layout is the A/B baseline, and nothing else in the fixture covers it now
        that the shared field defaults the pad ON.
      * ``mk_dispatch`` routes a request through the gfx942 dispatch factory
        (``_dense_spec``), so the emitted spec carries whatever the SHIPPED path folds in
        (for D64: the inherited K row-group pad + the bf16 ``waves_per_eu`` bump). These
        GUARD the IR that actually ships. Because they re-derive the spec from the
        dispatch policy rather than hard-coding the levers, a future D64 tuning change is
        captured automatically on the next re-bless -- no per-lever edits here."""
    from kernels.gfx942.attention_dense import (
        AttentionDenseSpec,
        build_attention_dense,
    )
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec

    base = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=16,
        num_kv_heads=4,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_n=64,
    )

    def mk(**over):
        d = dict(base)
        d.update(over)
        return lambda: build_attention_dense(AttentionDenseSpec(**d), arch=_ARCH)

    def mk_dispatch(**over):
        """Build via the gfx942 dispatch spec so the case tracks the SHIPPED config
        (incl. the inherited D64 K row-group pad and the ``waves_per_eu`` bump). Small
        Sq keeps the auto persistent decision on the default grid, matching the ``mk``
        D64 cases so the ONLY delta vs. the unpadded golden is the shipped pad."""
        req = AttentionRequest(
            batch=base["batch"],
            nhead_q=base["num_query_heads"],
            nhead_k=base["num_kv_heads"],
            seqlen_q=base["seqlen_q"],
            seqlen_k=base["seqlen_kv"],
            hdim_q=over.get("head_size", base["head_size"]),
            hdim_v=over.get("head_size", base["head_size"]),
            arch=_ARCH,
            mask_type=1 if over.get("causal", base["causal"]) else 0,
            dtype=over.get("dtype", base["dtype"]),
            algorithm="attention_dense",
        )
        return lambda: build_attention_dense(_dense_spec(req), arch=_ARCH)

    return {
        # --- default grid: dtype x head_size x causal/full x GQA/MHA x block_n ---
        "attention_dense_gfx942/default_d128_bf16_causal": mk(),
        "attention_dense_gfx942/default_d128_fp16_causal": mk(dtype="fp16"),
        # UNPADDED D64 (lds_k_group_pad=0): the A/B baseline layout. Explicit, because
        # the shared field defaults the pad ON -- without it nothing pins this codegen.
        "attention_dense_gfx942/default_d64_bf16_causal": mk(
            head_size=64, lds_k_group_pad=0
        ),
        "attention_dense_gfx942/default_d64_fp16_causal": mk(
            head_size=64, dtype="fp16", lds_k_group_pad=0
        ),
        # --- SHIPPED D64 path: dispatch inherits the shared K row-group pad (both
        #     dtypes) + the bf16 waves_per_eu bump. These guard the padded IR that
        #     actually ships; the unpadded default_d64_* cases above pin the other. ---
        "attention_dense_gfx942/dispatch_d64_bf16_causal": mk_dispatch(head_size=64),
        "attention_dense_gfx942/dispatch_d64_fp16_causal": mk_dispatch(
            head_size=64, dtype="fp16"
        ),
        "attention_dense_gfx942/default_d128_fp16_full": mk(dtype="fp16", causal=False),
        "attention_dense_gfx942/default_d128_bf16_mha": mk(
            num_query_heads=16, num_kv_heads=16
        ),
        "attention_dense_gfx942/default_d128_fp16_gqa16_8": mk(
            dtype="fp16", num_kv_heads=8
        ),
        # bn128 at D64: exercises the block_n=128 tiling; D64 keeps K_lds+V_lds
        # (2*128*64*2 = 32 KB) within the gfx942 64 KB budget (bn128 at D128 is 68 KB
        # -> correctly rejected, so it is not a golden case).
        "attention_dense_gfx942/default_d64_bf16_bn128": mk(head_size=64, block_n=128),
        # --- persistent (grid-stride) grid + both decodes ---
        "attention_dense_gfx942/persist_d128_bf16_qbmaj": mk(
            persistent=True, num_persistent=304, persist_decode="qb_major"
        ),
        "attention_dense_gfx942/persist_d128_fp16_hkvmaj": mk(
            dtype="fp16",
            persistent=True,
            num_persistent=304,
            persist_decode="hkv_major",
        ),
        "attention_dense_gfx942/persist_d128_bf16_intl": mk(
            persistent=True, num_persistent=304, interleave=True
        ),
        # D64 persistent locks the 2-rows/instr packed DMA loader on the persistent
        # builder (the default_d64 cases only exercise the default grid).
        "attention_dense_gfx942/persist_d64_fp16_qbmaj": mk(
            head_size=64,
            dtype="fp16",
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
        ),
        "attention_dense_gfx942/persist_d64_bf16_hkvmaj": mk(
            head_size=64,
            persistent=True,
            num_persistent=304,
            persist_decode="hkv_major",
        ),
    }


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc():
    doc = {"schema": "attention_dense_gfx942.ir_golden_sha256/v1", "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for cid, build in _cases().items():
            try:
                sha, nbytes = _sha_for(build, flavor)
                cases[cid] = {"sha256": sha, "bytes": nbytes}
            except Exception as e:  # pragma: no cover - diagnostic
                cases[cid] = {"error": str(e)[:160]}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


def test_attention_dense_gfx942_ir_matches_golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("gfx942 golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    gflav = golden.get("flavors", {}).get(flavor)
    if not gflav:
        pytest.skip(f"no gfx942 golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, build in _cases().items():
        want = gflav["cases"].get(cid, {}).get("sha256")
        if want is None:
            continue
        got, _ = _sha_for(build, flavor)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, "gfx942 attention_dense IR drift vs golden:\n  " + "\n  ".join(
        drift
    )


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_attention_dense_gfx942_ir_matches_golden()
