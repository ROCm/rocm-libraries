# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the softmax-gradient VALU levers of the backward.

Levers (``kernels.common._attention_bwd_valu``): ``bf16_cast``,
``bwd_exp2``, ``stats_fma``, ``stats_vec``.

* ``test_lever_agrees``: every lever value (and the stack of all of them) on
  bf16 / fp16, MHA / GQA / ``h_k != h_v``, odd lengths (stats tails), a
  ragged case, in every configuration of the arch: passes the float64 oracle;
  the bitwise levers (``rne_noguard``, ``stats_vec`` 2 / 4, also with edge
  tiles and the step-top stats loads) give dK / dV bit for bit equal to the
  default run in direct mode; a value the configuration cannot take is
  skipped with the reason;
* ``test_fast_exp2_flushes_only_below_2_pow_neg_126``: rows whose P is
  entirely below ``2^-126`` (lse2 raised by 140 after prep): under
  ``bwd_exp2 = "fast"`` their dQ is exactly 0, under ``"exact"`` it is
  subnormal-sized; every other row and dK / dV agree within the tolerance;
* ``test_rne_noguard_matches_fptrunc_on_every_f32``: a probe kernel rounds
  all 2^32 f32 patterns both ways on the device: zero differences on non-NaN
  inputs (the device's own ``fptrunc``, a conversion instruction on gfx950),
  and the NaN results are counted (kept NaN or not).

Selection: ``ROCKE_BWD_CONFIG`` / ``ROCKE_BWD_SHARD`` as in
``test_attention_bwd_numeric``. Skipped unless the device is gfx942, gfx950,
gfx1151 or gfx1201.
"""

from __future__ import annotations

import ctypes
import os
import struct

import numpy as np
import pytest

from ._attention_bwd_test_utils import find_bwd_case
from ._attention_bwd_harness import (
    ARCH,
    DEVICE_ARCHS,
    run_case,
    skip_unless_planned,
)
from .sdpa.bwd_cases import TOLERANCE_BWD, compare_bwd
from .sdpa.bwd_kernel_cases import selected_configs, shard_of

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

CASES = (
    "bwd_dtype_bf16_d128_causal_tl",
    "bwd_dtype_bf16_d64_mha",
    "bwd_lse_bf16_br_odd_lengths",
    "bwd_ragged_mqa_bf16_tl",
    "bwd_dtype_fp16_d128_mha",
    "bwd_lse_fp16_tl_odd_lengths",
    "bwd_gqa_gqa4_causal_tl",
    "bwd_kern_hkv_842_tl_fp16_d64",
)
# name -> (overrides, bitwise in direct mode, bf16 only)
VARIANTS = {
    "rne_noguard": ({"bf16_cast": "rne_noguard"}, True, True),
    "exp2_fast": ({"bwd_exp2": "fast"}, False, False),
    "stats_fma": ({"stats_fma": True}, False, False),
    "stats_vec2": ({"stats_vec": 2}, True, False),
    "stats_vec4": ({"stats_vec": 4}, True, False),
    "stats_vec4_edge_top": (
        {"stats_vec": 4, "edge_tiles": True, "stats_load": "step_top"},
        True,
        False,
    ),
    "stack": (
        {"bf16_cast": "rne_noguard", "bwd_exp2": "fast", "stats_fma": True,
         "stats_vec": 4},
        False,
        True,
    ),  # fmt: skip
    "stack_fp16": (
        {"bwd_exp2": "fast", "stats_fma": True, "stats_vec": 4},
        False,
        False,
    ),
}
_PLAIN: dict = {}


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


def _params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for cid in CASES:
        if shard_of(cid, n) != i:
            continue
        bf16 = find_bwd_case(cid).dtype == "bf16"
        for name, (_ov, _bit, bf16_only) in VARIANTS.items():
            if bf16_only and not bf16:
                continue
            if name == "stack_fp16" and bf16:
                continue
            for cfg in selected_configs(ARCH):
                out.append(pytest.param(cid, name, cfg, id=f"{cid}-{name}-{cfg}"))
    return out


def _require_legal(case, config, overrides):
    """Skip (with the reason) a lever the configuration's plan refuses: the
    case planned with ``overrides`` exactly as the run plans it (a refusal of
    the configuration alone is raised, not skipped)."""
    skip_unless_planned(case, arch=ARCH, config=config, overrides=overrides)


@pytest.mark.parametrize("case_id,variant,config", _params())
def test_lever_agrees(case_id, variant, config):
    case = find_bwd_case(case_id)
    overrides, bitwise, _ = VARIANTS[variant]
    _require_legal(case, config, overrides)
    key = (case_id, config)
    if key not in _PLAIN:
        base = run_case(case, arch=ARCH, config=config)
        _passes(base)
        _PLAIN[key] = (base.plan.specs["main"].dkv_mode, base.dk, base.dv)
    mode, dk, dv = _PLAIN[key]
    run = run_case(case, arch=ARCH, config=config, overrides=overrides)
    for k, v in overrides.items():
        assert getattr(run.plan.specs["main"], k) == v, k
    _passes(run)
    if bitwise and mode == "direct":
        assert np.array_equal(run.dk, dk), f"dK differs under {variant}"
        assert np.array_equal(run.dv, dv), f"dV differs under {variant}"


# ---------------------------------------------------------------------------
# fast exp2 on P below 2^-126
# ---------------------------------------------------------------------------

FAINT_CASES = ("bwd_dtype_bf16_d64_mha", "bwd_dtype_fp16_d128_mha")
_FAINT_LSE2_SHIFT = 140.0  # log2 units: every P of the row drops below 2^-126


def _faint_rows(case):
    """Bool ``[B, H_q, S_q]``: every third row of the first query head."""
    rows = np.zeros((case.b, case.h_q, case.s_q), bool)
    lq, _ = case.lens()
    for bi, n in enumerate(lq):
        rows[bi, 0, : int(n) : 3] = True
    return rows


def _raise_lse2(rows):
    def hook(ws, plan):
        main = plan.launches[1].scalars
        off, _ = plan.workspace_layout["WS_LSE2"]
        n_rows, s_q = main["ws_rows_q"], main["S_q_max"]
        lse2 = ws[off // 4 : off // 4 + n_rows * main["h_q"]]
        for bi, h, q in np.argwhere(rows):
            lse2[h * n_rows + bi * s_q + q] += np.float32(_FAINT_LSE2_SHIFT)

    return hook


def _shard():
    return tuple(int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))


def _faint_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = _shard()
    return [
        pytest.param(cid, cfg, id=f"{cid}-{cfg}")
        for cid in FAINT_CASES
        if shard_of(cid, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,config", _faint_params())
def test_fast_exp2_flushes_only_below_2_pow_neg_126(case_id, config):
    case = find_bwd_case(case_id)
    rows = _faint_rows(case)
    # a mode the configuration cannot take (a recorded resource refusal)
    # skips with the validator's reason
    for mode in ("exact", "fast"):
        skip_unless_planned(
            case, arch=ARCH, config=config, overrides={"bwd_exp2": mode}
        )
    runs = {}
    for mode in ("exact", "fast"):
        runs[mode] = run_case(
            case, arch=ARCH, config=config, overrides={"bwd_exp2": mode},
            after_prep=_raise_lse2(rows),
        )  # fmt: skip
        assert not runs[mode].integrity, runs[mode].integrity
    exact, fast = runs["exact"], runs["fast"]
    # dQ: [B, H, S, D]; the faint rows of head 0
    faint_fast = fast.dq[rows]
    faint_exact = exact.dq[rows]
    assert np.all(faint_fast == 0.0), "fast exp2 left a P below 2^-126 non-zero"
    assert np.all(np.abs(faint_exact) <= 2.0**-100), "faint rows are not tiny"
    tol = TOLERANCE_BWD[case.dtype]
    for name in ("dq", "dk", "dv"):
        a, b = getattr(exact, name), getattr(fast, name)
        if name == "dq":
            a, b = a[~rows], b[~rows]
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), name
        assert np.allclose(b, a, rtol=tol["rtol"], atol=tol["atol"]), name


# ---------------------------------------------------------------------------
# exhaustive guard-free rounding probe
# ---------------------------------------------------------------------------

_PROBE_THREADS = 256
_PROBE_BLOCKS = 1 << 22  # 2^22 blocks x 256 threads x 4 patterns = 2^32


def _make_cast_probe(arch):
    from rocke.core.ir import BF16, F32, I32, IRBuilder, PtrType, VectorType

    from kernels.common._attention_bwd_valu import rne_noguard_bf16

    b = IRBuilder(f"bwd_bf16_cast_probe_{arch}")
    cnt = b.param("CNT", PtrType(I32, "global"))
    b.kernel.attrs["max_workgroup_size"] = _PROBE_THREADS
    tid, bid = b.thread_id_x(), b.block_id_x()
    gid = b.add(b.mul(bid, b.const_i32(_PROBE_THREADS)), tid)
    w0 = b.shl(gid, b.const_i32(2))
    words = [b.lor(w0, b.const_i32(j)) for j in range(4)]
    f = b.vec_bitcast(b.vec_pack(words, I32), VectorType(F32, 4))
    gen = b.vec_bitcast(b.vec_cast_f32_to(f, BF16), VectorType(I32, 2))
    nog = b.vec_bitcast(rne_noguard_bf16(b, f), VectorType(I32, 2))
    one, zero = b.const_i32(1), b.const_i32(0)

    def half(v, j):
        w = b.vec_extract(v, j // 2)
        return (
            b.land(w, b.const_i32(0xFFFF)) if j % 2 == 0 else b.lshr(w, b.const_i32(16))
        )

    sums = [zero, zero, zero, zero]
    for j in range(4):
        is_nan = b.cmp_gt(
            b.land(words[j], b.const_i32(0x7FFFFFFF)), b.const_i32(0x7F800000)
        )
        g, n = half(gen, j), half(nog, j)
        diff = b.cmp_ne(g, n)
        kept = b.cmp_gt(b.land(n, b.const_i32(0x7FFF)), b.const_i32(0x7F80))
        conds = (
            b.land(diff, b.lnot(is_nan)),  # non-NaN mismatch: must stay 0
            is_nan,  # NaN inputs seen
            b.land(is_nan, kept),  # NaN that stays NaN without the guard
            b.land(is_nan, diff),  # NaN whose bits differ from fptrunc
        )
        sums = [b.add(s, b.select(c, one, zero)) for s, c in zip(sums, conds)]
    with b.scf_if(b.cmp_gt(b.add(sums[0], sums[1]), zero)):
        for i, s in enumerate(sums):
            b.global_atomic_add(cnt, b.const_i32(i), s)
    with b.scf_if(b.cmp_eq(tid, zero)):
        b.global_atomic_add(cnt, b.const_i32(4), one)  # blocks that ran
    b.ret()
    return b.kernel


def test_rne_noguard_matches_fptrunc_on_every_f32():
    if _shard()[0] != 0:
        pytest.skip("the exhaustive probe runs in shard 0 only")
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    art = compile_kernel(
        _make_cast_probe(ARCH), arch=ARCH, capture_ir_text=False, backend="python"
    )
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    counts = np.zeros(8, np.int32)
    buf = (ctypes.c_uint8 * counts.nbytes).from_buffer(counts)
    p = rt.alloc(counts.nbytes)
    try:
        rt.memcpy_h2d(p, buf, counts.nbytes)
        rt.launch(
            fn, (_PROBE_BLOCKS, 1, 1), (_PROBE_THREADS, 1, 1), struct.pack("<Q", p)
        )
        rt.sync()
        rt.memcpy_d2h(buf, p, counts.nbytes)
    finally:
        rt.free(p)
        mod.unload()
    mismatch, nans, kept, nan_diff, blocks = (int(x) for x in counts[:5])
    print(
        f"[bf16 cast probe {ARCH}] non-NaN mismatches={mismatch} NaN inputs={nans} "
        f"NaN kept NaN={kept} NaN bits differing={nan_diff}"
    )
    assert blocks == _PROBE_BLOCKS  # every pattern was visited
    assert nans == (1 << 24) - 2  # 2 signs x (2^23 - 1) mantissas
    assert mismatch == 0
    # some NaN payloads do not survive the guard-free rounding (-0 / +-Inf)
    assert 0 < kept < nans and nan_diff > 0
