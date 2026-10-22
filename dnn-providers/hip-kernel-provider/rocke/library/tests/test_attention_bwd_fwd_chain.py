# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Forward-to-backward chain of the general attention backward (device).

The backward consumes only the documented statistics contract: the stored O
and the natural-log fp32 LSE (``-inf`` on fully masked rows; rows past
``len_q`` are never written by the forward and hold garbage). These tests feed
it statistics that a forward kernel produced on the device instead of the
oracle's:

* the probe forward of the extension seam
  (``platform/tests/instances/_attention_fwd_ext_harness.make_probe_kernel``,
  inline natural-log LSE epilogue): dense BHSD forward, MHA / GQA, top-left
  and bottom-right causal, windows, fully masked rows, fp16 / bf16, d64 /
  d128. The backward is compared with the oracle evaluated on the **stored**
  O / LSE and with the oracle that recomputes LSE from Q and K;
* the same forward with the ``_lse_store`` epilogue
  (``library/tests/_lse_store_support.py``), run per batch at the batch's own
  lengths on padded cases: the LSE rows past ``len_q`` stay NaN and the
  backward must ignore them;
* a frozen fixture (``sdpa/bwd_fixture_v1.py``): Q, K, V, O, LSE and dO of
  one case, produced once by the in-tree forward and stored as text. Every
  later backward must keep passing it;
* negative controls on forward-produced statistics: the device LSE read as a
  log2 statistic must fail, and the main kernel built without the dead-row
  term of the dS select (``sentinel=False``) must give NaN where a dead row
  meets an ungated Dsum, while the shipped body keeps every gradient finite.

The forward runs on gfx942, gfx950 and gfx1151 (the probe shell's bodies); the
fixture test runs on every backward device arch. Configurations follow
``ROCKE_BWD_CONFIG``.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import math
import struct

import numpy as np
import pytest

from rocke.numeric.sdpa_reference import sdpa_reference_bwd

from ._attention_bwd_test_utils import device_configs
from ._attention_bwd_harness import (
    ARCH,
    DEVICE_ARCHS,
    dead_row_inputs,
    dead_rows,
    dec,
    enc,
    main_built_with,
    reference,
    run_case,
    ungated_dsum_hook,
)
from .sdpa import bwd_fixture_v1 as fixture
from .sdpa.bwd_cases import (
    SdpaBwdInputs,
    SdpaBwdReference,
    _mkb,
    _oracle_kwargs,
    _view,
    compare_bwd,
    evaluate_bwd,
    get_bwd_case,
    layout_tensors_bwd,
)

FWD_ARCHS = ("gfx942", "gfx950", "gfx1151")
LOG2E = np.float32(1.4426950408889634)
O_SENTINEL = 7.0
# |device LSE - oracle LSE| on finite rows: same rounded inputs, fp32 forward.
LSE_TOL = 2e-3

device = pytest.mark.skipif(
    ARCH not in DEVICE_ARCHS,
    reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
)
forward_device = pytest.mark.skipif(
    ARCH not in FWD_ARCHS,
    reason=f"the probe forward needs gfx942/gfx950/gfx1151; device is {ARCH}",
)

_FWD_KERNELS: dict = {}


# ---------------------------------------------------------------------------
# the forward on the device
# ---------------------------------------------------------------------------


def _forward_kernel(arch, dtype, d, group, lse_store):
    """The compiled probe forward and its kernel-argument list (cached)."""
    key = (arch, dtype, d, group, lse_store)
    if key in _FWD_KERNELS:
        return _FWD_KERNELS[key]
    from rocke.helpers.compile import compile_kernel

    from ._lse_store_support import factory_for  # also puts the probe on sys.path

    import _attention_fwd_ext_harness as probe  # noqa: E402

    name = f"bwd_chain_fwd_{arch}_{dtype}_d{d}_g{group}_{'store' if lse_store else 'inline'}"
    kernel, params = probe.make_probe_kernel(
        name,
        arch=arch,
        head_size=d,
        group=group,
        dtype="f16" if dtype == "fp16" else "bf16",
        lse_hook_factory=factory_for(arch) if lse_store else None,
    )
    art = compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")
    _FWD_KERNELS[key] = (art, params)
    return _FWD_KERNELS[key]


def device_forward(case, q, k, v, *, arch, lse_store=False, lse_fill=np.nan):
    """O and natural-log LSE of ``case`` computed on the device by the probe
    forward, one launch per batch at the batch's own lengths.

    ``q`` / ``k`` / ``v``: logical ``[B, H, S, D]`` values exact in the case
    dtype (rows past the lengths are not read: they are uploaded as NaN).
    Returns ``(o, lse)``: ``o`` ``[B, H_q, S_q, D]`` float32 (rows past
    ``len_q`` keep a sentinel) and ``lse`` ``[B, H_q, S_q]`` float32 (rows past
    ``len_q`` keep ``lse_fill``: the forward never writes them).
    """
    from rocke.core.arch import ArchTarget
    from rocke.runtime.hip_module import Runtime

    if case.h_k != case.h_v or case.h_q % case.h_k:
        raise ValueError("the probe forward needs h_k == h_v dividing h_q")
    dt, d, hq, hk = case.dtype, case.d, case.h_q, case.h_k
    sq, sk = case.s_q, case.s_kv
    art, params = _forward_kernel(arch, dt, d, hq // hk, lse_store)
    left, right, top_left = case.band
    lq, lk = case.lens()
    wave = ArchTarget.from_gfx(arch).wave_size
    o_all = np.full((case.b, hq, sq, d), O_SENTINEL, np.float32)
    lse_all = np.full((case.b, hq, sq), lse_fill, np.float32)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    try:
        for bi in range(case.b):
            if lq[bi] == 0:
                continue
            host = {}
            for name, x, n_rows, heads in (
                ("Q", q, lq[bi], hq),
                ("K", k, lk[bi], hk),
                ("V", v, lk[bi], hk),
            ):
                buf = np.full((heads, sq if name == "Q" else sk, d), np.nan, np.float32)
                buf[:, :n_rows] = x[bi, :, :n_rows]
                host[name] = enc(buf, dt)
            host["O"] = np.full(hq * sq * d, enc(np.float32(O_SENTINEL), dt)[()])
            host["LSE"] = np.full(hq * sq, lse_fill, np.float32)
            host["BIAS"] = np.zeros(4, np.float32)
            host["STATS"] = np.zeros(4, np.float32)
            dev = {}
            try:
                for name, a in host.items():
                    dev[name] = rt.alloc(max(int(a.nbytes), 16))
                    rt.memcpy_h2d(dev[name], _u8(a), a.nbytes)
                values = dict(
                    scale_log2=case.resolved_scale * math.log2(math.e),
                    seqlen_q=lq[bi], seqlen_k=lk[bi],
                    q_b=0, q_h=sq * d, q_t=d,
                    k_b=0, k_h=sk * d, k_t=d,
                    v_b=0, v_h=sk * d, v_t=d,
                    o_b=0, o_h=sq * d, o_t=d,
                    lse_b=0, lse_h=sq, stats_h=sq,
                    bias_b=0, bias_h=0, bias_q=0,
                    left=left, right=right,
                    diag=0 if top_left else lk[bi] - lq[bi],
                )  # fmt: skip
                packed = b""
                for name, fmt in params:
                    if fmt == "Q":
                        packed += struct.pack("<Q", dev[name])
                    else:
                        packed += struct.pack("<" + fmt, values[name])
                rt.launch(fn, ((lq[bi] + 15) // 16, hq, 1), (wave, 1, 1), packed)
                rt.sync()
                for name in ("O", "LSE"):
                    rt.memcpy_d2h(_u8(host[name]), dev[name], host[name].nbytes)
            finally:
                for p in dev.values():
                    rt.free(p)
            o_all[bi] = dec(host["O"], dt).reshape(hq, sq, d)
            lse_all[bi] = host["LSE"].reshape(hq, sq)
    finally:
        mod.unload()
    return o_all, lse_all


def _u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


# ---------------------------------------------------------------------------
# backward inputs from forward statistics
# ---------------------------------------------------------------------------


def chain_inputs(case, o, lse, *, base=None):
    """``((inputs, oracle), oracle_recomputed)`` of ``case`` with O and LSE
    replaced by ``o`` / ``lse`` on the valid rows (LSE: every row, garbage
    included). ``oracle`` runs on the stored values; ``oracle_recomputed``
    derives LSE from Q and K (same stored O). ``base``: the inputs whose Q, K,
    V and dO are used (default: the case's materialised inputs)."""
    inp = reference(case)[0] if base is None else base
    vq = inp.valid_q
    o_buf = np.array(inp.buffers["o"], np.float32, copy=True)
    view = _view(o_buf, inp.descs["o"])
    view[...] = np.where(vq[..., None], o, view)
    o_log = np.array(view, np.float64, copy=True)
    lse_dev = np.ascontiguousarray(lse, np.float32)
    dev = dataclasses.replace(
        inp,
        buffers={**inp.buffers, "o": o_buf, "lse": lse_dev},
        o=o_log,
        lse=lse_dev.astype(np.float64),
    )
    # the oracle ignores rows past len_q; keep the garbage out of its arithmetic
    stored = np.where(vq, lse_dev, 0.0).astype(np.float64)
    ref = evaluate_bwd(dataclasses.replace(dev, lse=stored))
    res = sdpa_reference_bwd(
        inp.q, inp.k, inp.v, o_log, inp.do, None, recompute_lse=True,
        **_oracle_kwargs(inp),
    )  # fmt: skip
    ref_re = SdpaBwdReference(
        dq=res.dq,
        dk=res.dk,
        dv=res.dv,
        delta=res.delta,
        valid_q=ref.valid_q,
        valid_kv=ref.valid_kv,
        dead_q=ref.dead_q,
    )
    return (dev, ref), ref_re


def _check_forward_stats(case, lse, base=None):
    """The device LSE matches the oracle forward LSE on the valid rows (same
    ``-inf`` pattern; finite rows within ``LSE_TOL``)."""
    inp = reference(case)[0] if base is None else base
    vq = inp.valid_q
    want = np.asarray(inp.lse, np.float64)[vq]
    got = np.asarray(lse, np.float64)[vq]
    assert np.array_equal(np.isneginf(got), np.isneginf(want))
    assert not np.isnan(got).any() and not np.isposinf(got).any()
    fin = np.isfinite(want)
    if fin.any():
        err = float(np.abs(got[fin] - want[fin]).max())
        assert err < LSE_TOL, err


def _check(run, ref=None, *, expect_ok=True):
    probs = compare_bwd(
        run.case, run.dq, run.dk, run.dv, run.ref if ref is None else ref
    )
    if expect_ok:
        assert not probs and not run.integrity, (probs, run.integrity)
    else:
        assert probs, "the negative control was not detected"


def _forward_run(case, *, lse_store=False):
    inp = reference(case)[0]
    return device_forward(case, inp.q, inp.k, inp.v, arch=ARCH, lse_store=lse_store)


# ---------------------------------------------------------------------------
# the probe forward (inline LSE epilogue)
# ---------------------------------------------------------------------------

# MHA / GQA, none / top-left / bottom-right / windows, dead rows, fp16 / bf16,
# d64 / d128, BHSD and BSHD backward layouts (the forward is BHSD).
PROBE_CASES = (
    "bwd_seqlen_33x72_br",
    "bwd_dtype_bf16_d128_mha",
    "bwd_headdim_bf16_d64_br_gqa",
    "bwd_gqa_gqa4_causal_tl",
    "bwd_window_symmetric_tl",
    "bwd_window_tl_q_gt_kv_fully_masked",
    "bwd_seqlen_100x17_br",
    "bwd_layout_bshd_gqa_br",
)


@forward_device
@pytest.mark.gpu
@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", PROBE_CASES)
def test_probe_forward_chain(case_id, config):
    case = get_bwd_case(case_id)
    o, lse = _forward_run(case)
    _check_forward_stats(case, lse)
    inputs, ref_re = chain_inputs(case, o, lse)
    run = run_case(case, arch=ARCH, config=config, inputs=inputs)
    _check(run)
    _check(run, ref_re)
    if run.ref.dead_q.any():
        assert np.all(run.dq[run.ref.dead_q] == 0.0)


# ---------------------------------------------------------------------------
# the _lse_store epilogue; rows past len_q are never written
# ---------------------------------------------------------------------------

LSE_STORE_CASES = (
    "bwd_causal_tl_eq_bf16",
    "bwd_padding_q_gt_kv_br",
    "bwd_padding_window_br",
    "bwd_padding_q_only_shorter",
)


@forward_device
@pytest.mark.gpu
@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
@pytest.mark.parametrize("case_id", LSE_STORE_CASES)
def test_lse_store_forward_chain(case_id, config):
    case = get_bwd_case(case_id)
    o, lse = _forward_run(case, lse_store=True)
    _check_forward_stats(case, lse)
    vq = reference(case)[0].valid_q
    # the forward left the rows past len_q alone: they still hold NaN
    assert np.isnan(lse[~vq]).all()
    if case.length_mode == "padded":
        assert (~vq).any()
    inputs, ref_re = chain_inputs(case, o, lse)
    run = run_case(case, arch=ARCH, config=config, inputs=inputs)
    _check(run)
    _check(run, ref_re)


# ---------------------------------------------------------------------------
# negative controls on forward-produced statistics
# ---------------------------------------------------------------------------

CONTROL_CASE = "bwd_seqlen_33x72_br"


@forward_device
@pytest.mark.gpu
@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
def test_chain_lse_in_log2_units_is_detected(config):
    case = get_bwd_case(CONTROL_CASE)
    o, lse = _forward_run(case)
    inputs, _ = chain_inputs(case, o, lse)
    _check(run_case(case, arch=ARCH, config=config, inputs=inputs))
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        inputs=inputs,
        mutate_inputs=lambda x: (x * LOG2E).astype(np.float32),
    )
    _check(run, expect_ok=False)


@forward_device
@pytest.mark.gpu
@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
def test_chain_dropped_dead_row_select_gives_nan(config):
    """Forward statistics with rows declared dead (``-inf`` LSE, NaN O) on
    band-live rows, and Dsum left ungated on them: the shipped body gives
    finite gradients (dQ of those rows exactly 0); the body without the
    ``lse2 < +inf`` term of the dS select lets the NaN through."""
    case = get_bwd_case(CONTROL_CASE)
    o, lse = _forward_run(case)
    (dev, _), _ = chain_inputs(case, o, lse)
    rows = dead_rows(case)
    inputs = dead_row_inputs(case, rows, [-np.inf], np.nan, base=dev)
    hook = ungated_dsum_hook(rows)
    run = run_case(case, arch=ARCH, config=config, inputs=inputs, after_prep=hook)
    _check(run)
    assert np.all(run.dq[rows] == 0.0)
    with main_built_with(sentinel=False) as kcache:
        bad = run_case(
            case, arch=ARCH, config=config, inputs=inputs, after_prep=hook,
            cache=kcache,
        )  # fmt: skip
    assert not np.isfinite(bad.dq).all() or not np.isfinite(bad.dk).all()


# ---------------------------------------------------------------------------
# the frozen fixture
# ---------------------------------------------------------------------------


def fixture_case():
    """The table-style case of the fixture (dense BHSD, batched, fixed)."""
    c = fixture.CASE
    return _mkb(
        f"fixture_v{fixture.VERSION}",
        reqs=(),
        tags=("fixture",),
        dtype=c["dtype"],
        b=c["b"],
        h_q=c["h_q"],
        h_k=c["h_k"],
        h_v=c["h_v"],
        s_q=c["s_q"],
        s_kv=c["s_kv"],
        d=c["d"],
        diagonal=c["diagonal"],
        left_bound=c["left_bound"],
        right_bound=c["right_bound"],
        fully_masked_rows=c["fully_masked_rows"],
    )


def fixture_inputs():
    """``((inputs, oracle), arrays)``: the fixture's stored tensors as backward
    inputs (contiguous BHSD buffers) and the oracle on exactly those values."""
    case = fixture_case()
    arr = fixture.load()
    descs, sizes = layout_tensors_bwd(case)
    buffers, logical = {}, {}
    for role in ("q", "k", "v", "o", "do"):
        dsc = descs[role]
        buf = np.full(sizes[dsc.buffer], np.nan, np.float32)
        _view(buf, dsc)[...] = arr[role].astype(np.float32)
        buffers[dsc.buffer] = buf
        logical[role] = arr[role].astype(np.float64)
    lse = np.ascontiguousarray(arr["lse"], np.float32)
    buffers["lse"] = lse
    fwd = _FixtureForward(
        q=logical["q"],
        k=logical["k"],
        v=logical["v"],
        scale=case.resolved_scale,
    )
    vq = np.ones((case.b, case.h_q, case.s_q), bool)
    vk = np.ones((case.b, case.s_kv), bool)
    inp = SdpaBwdInputs(
        case=case,
        fwd=fwd,
        buffers=buffers,
        descs=descs,
        sizes=sizes,
        o=logical["o"],
        do=logical["do"],
        lse=lse.astype(np.float64),
        valid_q=vq,
        valid_kv=vk,
    )
    return (inp, evaluate_bwd(inp)), arr


@dataclasses.dataclass
class _FixtureForward:
    """The forward fields the backward oracle and harness read."""

    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    scale: float
    seq_len_q: None = None
    seq_len_kv: None = None
    q_offsets: None = None
    kv_offsets: None = None


def test_frozen_fixture_is_intact():
    arr = fixture.load()
    c = fixture.CASE
    shapes = {
        "q": (c["b"], c["h_q"], c["s_q"], c["d"]),
        "k": (c["b"], c["h_k"], c["s_kv"], c["d"]),
        "v": (c["b"], c["h_v"], c["s_kv"], c["d"]),
        "o": (c["b"], c["h_q"], c["s_q"], c["d"]),
        "do": (c["b"], c["h_q"], c["s_q"], c["d"]),
        "lse": (c["b"], c["h_q"], c["s_q"]),
    }
    assert {r: a.shape for r, a in arr.items()} == shapes
    digest = hashlib.sha256()
    for role in fixture.ROLES:
        digest.update(np.ascontiguousarray(arr[role]).tobytes())
    assert digest.hexdigest() == fixture.SHA256
    # natural-log LSE with fully masked rows, as the forward stores it
    lse = arr["lse"]
    assert np.isneginf(lse).any() and np.isfinite(lse).any()
    assert not np.isnan(lse).any() and not np.isposinf(lse).any()
    assert np.all(arr["o"][np.isneginf(lse)] == 0)
    (_, ref), _ = fixture_inputs()
    assert ref.dead_q.sum() == np.isneginf(lse).sum()
    assert np.isfinite(ref.dq).all() and np.isfinite(ref.dk).all()


@device
@pytest.mark.gpu
@pytest.mark.parametrize("config", device_configs(ARCH, DEVICE_ARCHS))
def test_frozen_fixture(config):
    inputs, _ = fixture_inputs()
    case = inputs[0].case
    run = run_case(case, arch=ARCH, config=config, inputs=inputs)
    _check(run)
    assert np.all(run.dq[run.ref.dead_q] == 0.0)


@forward_device
@pytest.mark.gpu
def test_current_forward_reproduces_the_fixture():
    """The forward of this tree still gives the fixture's O and LSE (within
    rounding: the fixture may come from another arch)."""
    (inp, _), arr = fixture_inputs()
    case = inp.case
    o, lse = device_forward(case, inp.q, inp.k, inp.v, arch=ARCH)
    stored = arr["lse"].astype(np.float64)
    assert np.array_equal(np.isneginf(lse), np.isneginf(stored))
    fin = np.isfinite(stored)
    assert float(np.abs(lse[fin] - stored[fin]).max()) < LSE_TOL
    ulp = 2.0**-10 if case.dtype == "fp16" else 2.0**-7
    err = np.abs(o.astype(np.float64) - arr["o"].astype(np.float64))
    assert float(err.max()) <= 2 * ulp * max(1.0, float(np.abs(arr["o"]).max()))
