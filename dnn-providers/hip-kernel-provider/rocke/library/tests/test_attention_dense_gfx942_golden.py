# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the gfx942 dense flash-attn kernel.

Hashes the Python-lowered LLVM IR (SHA256) of representative gfx942 ``attention_dense``
specs and compares against a checked-in per-flavor golden fixture, catching any
unintended codegen drift across the P1-P4 lever set. Pure text lowering -- no GPU / no
comgr required.

Every flavor in ``LLVM_FLAVORS`` is checked from any host, through the shared
``rocke.core.ir_golden`` comparator, so a newly added flavor fails here until the
fixture is re-blessed.

Covers the acceptance matrix: D64/D128 x bf16/fp16 x default/persistent x GQA, causal
and full, plus sliding-window (W128/W256, both grids, D64/D128) and sinks (alone and
combined). Every case is in the gfx942 supported set; varlen and ragged are rejected on
gfx942, so unlike the gfx950 sibling they are absent here. The opt-in bf16 D128
K-double-buffer (``lds_num_buffers=2``) and V-register-prefetch
(``v_reg_prefetch=True``) knobs are pinned individually and together on supported
persistent grids (including LSE); no shipped default reaches the new pipeline paths.

Both D64 K-LDS layouts are pinned so drift on either is caught:
  * ``default_d64_*``  -- specs built DIRECTLY with ``lds_k_group_pad=0``: the UNPADDED
    layout, i.e. the A/B baseline the pad's ~2x is measured against.
  * ``dispatch_d64_*`` -- specs built through the gfx942 dispatch factory
    (``attention_tuning_spec(req, "gfx942_dense")``), which inherits the shared default pad -> this GUARDS the padded
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
    while covering CFVST on the default grid and KDB/VPF on persistent decodes.

    Two families of builders:
      * ``mk`` builds an ``AttentionDenseSpec`` DIRECTLY. The D64 cases pass
        ``lds_k_group_pad=0`` explicitly, pinning the UNPADDED K layout -- keep them:
        that layout is the A/B baseline, and nothing else in the fixture covers it now
        that the shared field defaults the pad ON.
      * ``mk_dispatch`` routes a request through the gfx942 dispatch factory
        (``attention_tuning_spec(req, "gfx942_dense")``), so the emitted spec carries whatever the SHIPPED path folds in
        (for D64: the inherited K row-group pad + the bf16 ``waves_per_eu`` bump). These
        GUARD the IR that actually ships. Because they re-derive the spec from the
        dispatch policy rather than hard-coding the levers, a future D64 tuning change is
        captured automatically on the next re-bless -- no per-lever edits here."""
    from kernels.gfx942.attention_dense import (
        AttentionDenseSpec,
        Gfx942AttentionDenseSpec,
        build_attention_dense,
    )
    from dispatch.attention import AttentionRequest, attention_tuning_spec

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
        gfx942_only = (
            d.get("emit_lse", False)
            or d.get("window_wave_skip", False)
            or any(k in d for k in ("lds_num_buffers", "v_reg_prefetch"))
        )
        spec_type = Gfx942AttentionDenseSpec if gfx942_only else AttentionDenseSpec
        return lambda: build_attention_dense(spec_type(**d), arch=_ARCH)

    def mk_gfx942(**over):
        d = dict(base)
        d.update(over)
        return lambda: build_attention_dense(Gfx942AttentionDenseSpec(**d), arch=_ARCH)

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
            sliding_window=over.get("sliding_window", 0),
            use_sinks=over.get("use_sinks", False),
            emit_lse=over.get("emit_lse", False),
        )
        return lambda: build_attention_dense(
            attention_tuning_spec(req, "gfx942_dense").kernel_spec, arch=_ARCH
        )

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
        # --- sliding-window (SWA): start_tile KV-loop prune + in-tile lower mask.
        #     W % block_n == 0 (128, 256 = 2, 4 tiles at block_n=64). Default and
        #     persistent grids both pruned per work item; dispatch case guards the
        #     gfx942_dense sliding_window threading end to end. ---
        "attention_dense_gfx942/swa_d128_bf16_w128": mk(sliding_window=128),
        "attention_dense_gfx942/swa_d128_fp16_w256": mk(
            dtype="fp16", sliding_window=256
        ),
        "attention_dense_gfx942/persist_swa_d128_bf16_w128": mk(
            sliding_window=128,
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
        ),
        "attention_dense_gfx942/dispatch_swa_d128_bf16_w128": mk_dispatch(
            sliding_window=128
        ),
        # Window spanning the query block (W >= block_m): three-phase band loop
        # (window-bound / mask-free / causal-bound) on the persistent grid, and the
        # narrow-band dispatch geometry for D64 bf16 (64-row blocks).
        "attention_dense_gfx942/persist_swa_d128_bf16_w256": mk(
            sliding_window=256,
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
        ),
        "attention_dense_gfx942/dispatch_swa_sink_d64_bf16_w128": mk_dispatch(
            head_size=64, sliding_window=128, use_sinks=True
        ),
        # Wave-uniform skip of fully masked (wave, tile) pairs on the persistent grid
        # (the dispatch D128 case above covers the one-CTA-per-block grid).
        "attention_dense_gfx942/persist_swa_wskip_d128_fp16_w128": mk(
            dtype="fp16",
            sliding_window=128,
            block_m=128,
            block_n=32,
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
            window_wave_skip=True,
        ),
        # D64 SWA: structurally distinct builder arm (packed 2-rows/DMA + K row-group
        # pad + wpe=4 tune). Numeric coverage is in _SWA_COHORT
        "attention_dense_gfx942/swa_d64_bf16_w128": mk(
            head_size=64, sliding_window=128
        ),
        # --- sinks: m/l seeded from sink_ptr[hq] once per work item, before the KV
        #     loop. Default and persistent grids (both decodes), full and causal,
        #     both head sizes, SWA-sink, and a dispatch case that guards the
        #     gfx942_dense._base_spec use_sinks threading end to end. ---
        "attention_dense_gfx942/sink_d128_bf16_causal": mk(use_sinks=True),
        "attention_dense_gfx942/sink_d128_fp16_full": mk(
            dtype="fp16", causal=False, use_sinks=True
        ),
        "attention_dense_gfx942/sink_d64_fp16_causal": mk(
            head_size=64, dtype="fp16", lds_k_group_pad=0, use_sinks=True
        ),
        "attention_dense_gfx942/persist_sink_d128_bf16_qbmaj": mk(
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
            use_sinks=True,
        ),
        "attention_dense_gfx942/persist_sink_d128_fp16_hkvmaj": mk(
            dtype="fp16",
            persistent=True,
            num_persistent=304,
            persist_decode="hkv_major",
            use_sinks=True,
        ),
        "attention_dense_gfx942/swa_sink_d128_bf16_w128": mk(
            sliding_window=128, use_sinks=True
        ),
        "attention_dense_gfx942/persist_swa_sink_d128_bf16_w128": mk(
            sliding_window=128,
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
            use_sinks=True,
        ),
        "attention_dense_gfx942/dispatch_sink_d64_bf16_causal": mk_dispatch(
            head_size=64, use_sinks=True
        ),
        # --- optional LSE output: ordinary epilogue, the persistent epilogue with
        #     keyless rows (window shorter than the distance to the last key), and the
        #     dispatcher path (AttentionRequest.emit_lse -> spec). ---
        "attention_dense_gfx942/lse_d128_bf16_causal": mk(emit_lse=True),
        "attention_dense_gfx942/lse_persist_empty_window_d128_bf16": mk(
            emit_lse=True,
            persistent=True,
            num_persistent=304,
            seqlen_q=256,
            seqlen_kv=64,
            sliding_window=64,
        ),
        "attention_dense_gfx942/dispatch_lse_d128_bf16_causal": mk_dispatch(
            emit_lse=True
        ),
        # --- sinks + LSE: a keyless row keeps the sink seed (m = sink * LOG2E,
        #     l = 1), so its LSE is the sink logit rather than -inf. Same geometry
        #     as the LSE empty-window case. ---
        # --- bottom-right causal (runtime diagonal offset) on both grids, the
        #     diagonal split, the KV-segment split, and the dispatcher path. ---
        "attention_dense_gfx942/br_d128_bf16": mk(
            seqlen_kv=1024, causal_bottom_right=True
        ),
        "attention_dense_gfx942/br_persist_d128_bf16": mk(
            seqlen_kv=1024,
            causal_bottom_right=True,
            persistent=True,
            num_persistent=304,
        ),
        "attention_dense_gfx942/br_dsplit_d64_bf16": mk_gfx942(
            head_size=64,
            seqlen_kv=1024,
            causal_bottom_right=True,
            causal_diag_split=True,
        ),
        "attention_dense_gfx942/br_dsplit_kvs4_d128_bf16": mk_gfx942(
            seqlen_kv=2048,
            causal_bottom_right=True,
            causal_diag_split=True,
            kv_segments=4,
            block_m=128,
        ),
        "attention_dense_gfx942/lse_sink_persist_empty_window_d128_bf16": mk(
            emit_lse=True,
            use_sinks=True,
            persistent=True,
            num_persistent=304,
            seqlen_q=256,
            seqlen_kv=64,
            sliding_window=64,
        ),
        # --- Opt-in K double buffering and V register prefetch on the BF16 D128
        #     CFVST body (on by default), persistent grid only. Pin both persistent
        #     decodes and LSE; V LDS stays single-buffered. ---
        "attention_dense_gfx942/persist_bthkvmin_kdb_d128_bf16_causal": mk(
            lds_num_buffers=2,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        "attention_dense_gfx942/persist_bthkvmin_vpf_d128_bf16_causal": mk(
            v_reg_prefetch=True,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        "attention_dense_gfx942/persist_bthkvmin_kdbvpf_d128_bf16_causal": mk(
            lds_num_buffers=2,
            v_reg_prefetch=True,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        "attention_dense_gfx942/persist_kdb_d128_bf16_qbmaj": mk(
            lds_num_buffers=2,
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
        ),
        "attention_dense_gfx942/persist_vpf_d128_bf16_qbmaj": mk(
            v_reg_prefetch=True,
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
        ),
        "attention_dense_gfx942/persist_kdbvpf_d128_bf16_qbmaj": mk(
            lds_num_buffers=2,
            v_reg_prefetch=True,
            persistent=True,
            num_persistent=304,
            persist_decode="qb_major",
        ),
        # The last KV tile is folded in after the loop; the LSE epilogue must
        # use the final row max independently for both pipeline levers.
        "attention_dense_gfx942/lse_persist_bthkvmin_kdb_d128_bf16": mk(
            emit_lse=True,
            lds_num_buffers=2,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        "attention_dense_gfx942/lse_persist_bthkvmin_vpf_d128_bf16": mk(
            emit_lse=True,
            v_reg_prefetch=True,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        "attention_dense_gfx942/lse_persist_bthkvmin_kdbvpf_d128_bf16": mk(
            emit_lse=True,
            lds_num_buffers=2,
            v_reg_prefetch=True,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        # The measured final optimization: KDB + VPF + K pad4 + diagonal split.
        "attention_dense_gfx942/persist_bthkvmin_kdbvpf_pad4_dsplit": mk(
            lds_num_buffers=2,
            v_reg_prefetch=True,
            lds_row_pad=4,
            causal_diag_split=True,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        # Diagonal split with each pipeline lever alone: each flag-on path has its
        # own pin.
        "attention_dense_gfx942/persist_bthkvmin_kdb_dsplit_d128_bf16": mk(
            lds_num_buffers=2,
            causal_diag_split=True,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
        "attention_dense_gfx942/persist_bthkvmin_vpf_dsplit_d128_bf16": mk(
            v_reg_prefetch=True,
            causal_diag_split=True,
            persistent=True,
            num_persistent=304,
            persist_decode="bt_hkv_minor",
        ),
    }


def _sha_for(build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _run(flavor):
    """One flavor's golden sub-document. A build error propagates with its
    traceback, in the test and under ``--write`` alike, so a fixture can never be
    blessed with a case that fails to lower."""
    cases = {}
    for cid, build in _cases().items():
        sha, nbytes = _sha_for(build, flavor)
        cases[cid] = {"sha256": sha, "bytes": nbytes}
    return {"cases": cases}


def _build_doc():
    from rocke.core.ir_golden import GOLDEN_ALIASES, GOLDEN_PROFILE_FLAVORS

    return {
        "schema": "attention_dense_gfx942.ir_golden_sha256/v1",
        "aliases": GOLDEN_ALIASES,
        "flavors": {fl: _run(fl) for fl in GOLDEN_PROFILE_FLAVORS},
    }


def test_attention_dense_gfx942_ir_matches_golden():
    import pytest
    from rocke.core.ir_golden import check_golden

    if not _GOLDEN.exists():
        pytest.skip("gfx942 golden fixture missing; generate with --write")
    drift = check_golden(_GOLDEN, _run)
    assert not drift, "gfx942 attention_dense IR drift vs golden:\n  " + "\n  ".join(
        drift
    )


def test_pipeline_golden_entries_cover_each_independent_mode_and_llvm_flavor():
    """Pin non-vacuous fixture coverage without generating unverified new hashes.

    This checks the *existing* golden matrix (including the pad4 + diagonal-split
    winner) independently of lowering. The main golden test separately verifies
    that the current lowered LLVM IR matches each recorded digest.
    """
    from rocke.core.ir_golden import GOLDEN_ALIASES, GOLDEN_PROFILE_FLAVORS

    doc = json.loads(_GOLDEN.read_text())
    expected = set(_cases())
    required = {
        "attention_dense_gfx942/persist_bthkvmin_kdb_d128_bf16_causal",
        "attention_dense_gfx942/persist_bthkvmin_vpf_d128_bf16_causal",
        "attention_dense_gfx942/persist_bthkvmin_kdbvpf_d128_bf16_causal",
        "attention_dense_gfx942/persist_kdb_d128_bf16_qbmaj",
        "attention_dense_gfx942/persist_vpf_d128_bf16_qbmaj",
        "attention_dense_gfx942/persist_kdbvpf_d128_bf16_qbmaj",
        "attention_dense_gfx942/lse_persist_bthkvmin_kdb_d128_bf16",
        "attention_dense_gfx942/lse_persist_bthkvmin_vpf_d128_bf16",
        "attention_dense_gfx942/lse_persist_bthkvmin_kdbvpf_d128_bf16",
        "attention_dense_gfx942/persist_bthkvmin_kdbvpf_pad4_dsplit",
        "attention_dense_gfx942/persist_bthkvmin_kdb_dsplit_d128_bf16",
        "attention_dense_gfx942/persist_bthkvmin_vpf_dsplit_d128_bf16",
    }
    assert required <= expected, sorted(required - expected)
    assert set(doc["flavors"]) == set(GOLDEN_PROFILE_FLAVORS)
    assert doc["aliases"] == GOLDEN_ALIASES
    for flavor, value in doc["flavors"].items():
        cases = value["cases"]
        assert set(cases) == expected, (
            f"{flavor}: missing={sorted(expected - set(cases))}, "
            f"extra={sorted(set(cases) - expected)}"
        )
        for case_id in required:
            entry = cases[case_id]
            assert len(entry["sha256"]) == 64 and entry["bytes"] > 0, (
                flavor,
                case_id,
                entry,
            )


def test_independent_pipeline_diag_split_changes_lowered_ir():
    """CPU codegen proof for each split-mode pipeline branch.

    Split-only KDB/VPF lowerings are compared with their unsplit twins after
    normalizing the symbol, so the split must change the pipeline body itself;
    each split mode is also pinned by its own golden SHA-256 entry.
    """
    from kernels.gfx942.attention_dense import (
        Gfx942AttentionDenseSpec,
        build_attention_dense,
        supports_attention_dense,
    )
    from rocke.core.lower_llvm import (
        _lower_kernel_to_llvm_python,
        _resolve_llvm_flavor,
    )

    base = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=16,
        num_kv_heads=4,
        head_size=128,
        dtype="bf16",
        causal=True,
        block_n=64,
        persistent=True,
        num_persistent=304,
        persist_decode="bt_hkv_minor",
    )
    fingerprints = {}
    for nbuf, vpf, label in (
        (2, False, "kdb"),
        (1, True, "vpf"),
        (2, True, "kdb_vpf"),
    ):
        fingerprints[label] = {}
        for split in (False, True):
            spec = Gfx942AttentionDenseSpec(
                **base,
                lds_num_buffers=nbuf,
                v_reg_prefetch=vpf,
                causal_diag_split=split,
            )
            ok, why = supports_attention_dense(spec, arch=_ARCH)
            assert ok, f"{label} split={split}: {why}"
            kd = build_attention_dense(spec, arch=_ARCH)
            llvm = _lower_kernel_to_llvm_python(
                kd, arch=_ARCH, llvm_flavor=_resolve_llvm_flavor()
            )
            # Exclude symbol renaming: require a genuine change to the IR body.
            normalized = llvm.replace(kd.name, "KERNEL")
            fingerprints[label][split] = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()
        assert (
            fingerprints[label][True] != fingerprints[label][False]
        ), f"{label} split and unsplit unexpectedly emit identical IR"
    assert (
        len({fingerprints[mode][True] for mode in fingerprints}) == 3
    ), "independent split-mode pipeline IRs unexpectedly aliased"


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_attention_dense_gfx942_ir_matches_golden()
