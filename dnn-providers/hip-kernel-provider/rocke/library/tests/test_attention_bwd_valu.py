# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the softmax-gradient VALU levers of the attention backward.

The levers (``kernels.common._attention_bwd_valu``): ``bf16_cast``,
``bwd_exp2``, ``stats_fma`` and ``stats_vec``; each default emits the shipped
kernel.

* spec: value sets and couplings (``rne_noguard`` needs bf16, ``stats_vec``
  needs the head-major workspace and an MFMA target), names salted, the
  explicit defaults are the default spec and IR;
* registry: one axis each, numerics class, per-arch sweep values;
* the guard-free bf16 rounding evaluated on host for every 16-bit upper half
  with a set of lower halves (subnormals, infinities and the NaN patterns of
  the docstring included) against round to nearest even: bit for bit for
  every non-NaN value, the documented results for NaN;
* the batched lse2 / Dsum loads evaluated on host against the scalar loads of
  ``StatsView.rows`` over workspace shapes with every tail case (the window
  inside a head, crossing into the previous or next head, rows past
  ``ws_rows``, heads shorter than the window, arrays shorter than the window
  inside the plan's rounded sub-buffer): the same values bit for bit, every
  access inside the buffer;
* IR: each lever changes exactly the ops it claims (casts, exp2, fma, the
  stats loads).
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from functools import cache

import numpy as np
import pytest
from rocke.core.ir import BF16
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from dispatch.attention_bwd import tuning_common as tc
from kernels.common._attention_bwd_addr import WorkspaceAddr
from kernels.common._attention_bwd_body import StatsView
from kernels.common._attention_bwd_valu import (
    BF16_CASTS,
    BWD_EXP2S,
    NATIVE_BF16_CVT_ARCHS,
    STATS_VECS,
    batched_stats_rows,
    cast_c_frag,
    check_stats_vec_layout,
    rne_noguard_bf16,
)
from kernels.common.attention_bwd import (
    BWD_ARCHS,
    AttnBwdSpec,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_plan import (
    AttnBwdRequest,
    attn_bwd_workspace_bytes,
    attn_bwd_workspace_layout,
)

from ._attention_bwd_int_eval import TV, IntEvalBuilder
from .sdpa.bwd_kernel_cases import config_knobs

LEVERS = {
    "bf16_cast": BF16_CASTS,
    "bwd_exp2": BWD_EXP2S,
    "stats_fma": (False, True),
    "stats_vec": STATS_VECS,
}
MFMA_ARCHS = tuple(a for a in BWD_ARCHS if attn_bwd_arch_facts(a).matrix_path == "mfma")


# ---------------------------------------------------------------------------
# spec and registry
# ---------------------------------------------------------------------------


def test_defaults_and_value_sets():
    spec = AttnBwdSpec(head_size=64, dtype="bf16")
    for name, values in LEVERS.items():
        assert getattr(spec, name) == values[0], name
        for v in values[1:]:
            other = dataclasses.replace(spec, **{name: v})
            assert (name, v) in other.non_default_fields()
            assert other.kernel_name("main") != spec.kernel_name("main")
    explicit = AttnBwdSpec(
        head_size=64, dtype="bf16", bf16_cast="generic", bwd_exp2="exact",
        stats_fma=False, stats_vec=1,
    )  # fmt: skip
    assert explicit == spec
    assert explicit.non_default_fields() == (("dtype", "bf16"),)
    for bad in (
        dict(bf16_cast="rtz"),
        dict(bwd_exp2="guarded"),
        dict(stats_vec=3),
        dict(stats_vec=True),
        dict(stats_vec="4"),
        dict(stats_fma=1),
    ):
        with pytest.raises(ValueError):
            AttnBwdSpec(head_size=64, dtype="bf16", **bad)


def test_couplings():
    with pytest.raises(ValueError, match="bf16"):
        AttnBwdSpec(head_size=64, dtype="fp16", bf16_cast="rne_noguard")
    with pytest.raises(ValueError, match="head_major"):
        AttnBwdSpec(head_size=64, ws_layout="token_major", stats_vec=4)
    for arch in BWD_ARCHS:
        spec = AttnBwdSpec(head_size=64, dtype="bf16", stats_vec=4)
        if arch in MFMA_ARCHS:
            assert validate_attn_bwd_spec(spec, arch).stats_vec == 4
        else:
            with pytest.raises(ValueError, match="MFMA"):
                validate_attn_bwd_spec(spec, arch)
        for kw in (
            dict(bf16_cast="rne_noguard"),
            dict(bwd_exp2="fast"),
            dict(stats_fma=True),
        ):
            validate_attn_bwd_spec(AttnBwdSpec(head_size=64, dtype="bf16", **kw), arch)


@pytest.mark.parametrize("arch", MFMA_ARCHS)
def test_mfma_c_rows_are_runs_of_four(arch):
    """The MFMA C map gives each lane runs of four consecutive rows (the fact
    ``stats_vec`` relies on); a C map without such runs is refused."""
    from kernels.common._attention_bwd_caps import select_bwd_atom

    for k in (16, 32):
        try:
            op = select_bwd_atom(arch, "bf16", k)
        except ValueError:
            continue
        for width in STATS_VECS:
            check_stats_vec_layout(op, width)
    wmma = select_bwd_atom("gfx1151", "bf16", 16)
    with pytest.raises(ValueError, match="consecutive"):
        check_stats_vec_layout(wmma, 2)


def test_registry_axes():
    for name, values in LEVERS.items():
        axis = tc.backward_axis(name)
        assert axis.choices == tuple(values) and axis.default == values[0]
        assert axis.family == 6 and axis.stage == "codegen"
        assert not axis.not_built and not axis.gated and not axis.plan_owned
    assert tc.backward_axis("bf16_cast").numerics == "bitwise"
    assert tc.backward_axis("stats_vec").numerics == "bitwise"
    assert tc.backward_axis("bwd_exp2").numerics == "reorders_fp"
    assert tc.backward_axis("stats_fma").numerics == "reorders_fp"
    # enumerated on the MFMA targets (the device-verified ones); the guard-free
    # rounding not where the generic cast is one conversion instruction
    for arch in BWD_ARCHS:
        mfma = arch in MFMA_ARCHS
        casts = tc.backward_axis("bf16_cast").values(arch)
        native = arch in NATIVE_BF16_CVT_ARCHS
        assert casts == (BF16_CASTS if mfma and not native else BF16_CASTS[:1])
        for name, values in LEVERS.items():
            if name == "bf16_cast":
                continue
            got = tc.backward_axis(name).values(arch)
            assert got == (tuple(values) if mfma else tuple(values[:1])), name
    cast = tc.backward_axis("bf16_cast")
    assert cast.applies("rne_noguard", {"dtype": "bf16"})
    assert not cast.applies("rne_noguard", {"dtype": "fp16"})
    vec = tc.backward_axis("stats_vec")
    assert vec.applies(4, {"ws_layout": "head_major"})
    assert not vec.applies(4, {"ws_layout": "token_major"})


# ---------------------------------------------------------------------------
# the guard-free bf16 rounding on host
# ---------------------------------------------------------------------------


def _rne_reference(u: np.ndarray) -> np.ndarray:
    """bf16 bits of the f32 bits ``u``: round to nearest even on the wrapped
    32-bit word (the guard-free formula, defined for every pattern)."""
    u = u.astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF) >> 16).astype(np.uint32)


def _generic_reference(u: np.ndarray) -> np.ndarray:
    """``fptrunc`` f32 -> bf16 bits: round to nearest even, NaN quieted."""
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    return np.where(nan, ((u | 0x400000) >> 16) & 0xFFFF, _rne_reference(u))


def _f32_vec(words) -> TV:
    vals = np.asarray(words, np.uint32).view(np.float32)
    return TV(tuple(vals[i] for i in range(len(vals))), f"vec<f32x{len(vals)}>")


def _bits_of(tv: TV) -> list[int]:
    return [int(np.asarray(x, np.float32).view(np.uint32)) >> 16 for x in tv.v]


def _patterns() -> np.ndarray:
    """Every 16-bit upper half with lower halves at the rounding boundaries."""
    hi = np.arange(1 << 16, dtype=np.uint32) << 16
    lows = np.array([0, 1, 0x7FFF, 0x8000, 0x8001, 0xC000, 0xFFFF], np.uint32)
    return (hi[:, None] | lows[None, :]).reshape(-1)


def test_rne_noguard_matches_round_to_nearest_even():
    u = _patterns()
    b = IntEvalBuilder()
    got = []
    for i in range(0, len(u), 4):
        got += _bits_of(rne_noguard_bf16(b, _f32_vec(u[i : i + 4])))
    got = np.array(got, np.uint32)
    assert np.array_equal(got, _rne_reference(u))
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    # every non-NaN pattern (+-0, subnormals, normals, +-Inf, the values that
    # round up to +-Inf) is the generic fptrunc bit for bit
    assert np.array_equal(got[~nan], _generic_reference(u)[~nan])
    assert nan.any() and (~nan).sum() > 400_000


@pytest.mark.parametrize(
    "u,want",
    [
        (0x7FFFFFFF, 0x8000),  # NaN -> -0 (the carry wraps the word)
        (0x7F800001, 0x7F80),  # signalling NaN -> +Inf
        (0xFF800001, 0xFF80),  # signalling NaN -> -Inf
        (0x7FC00000, 0x7FC0),  # the quiet NaN arithmetic produces stays NaN
        (0xFFC00000, 0xFFC0),
        (0x7F7FFFFF, 0x7F80),  # largest finite rounds to +Inf (as fptrunc)
        (0x00000001, 0x0000),  # smallest subnormal rounds to +0
        (0x00008000, 0x0000),  # tie to even
        (0x00018000, 0x0002),  # tie to even (odd lsb rounds up)
        (0x80000000, 0x8000),
    ],
)
def test_rne_noguard_documented_results(u, want):
    b = IntEvalBuilder()
    got = _bits_of(rne_noguard_bf16(b, _f32_vec([u, u, u, u])))
    assert got == [want] * 4
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    if not nan:
        assert want == int(_generic_reference(np.array([u], np.uint32))[0])


def test_cast_dispatch():
    b = IntEvalBuilder()
    v = _f32_vec([0x3F800000, 0x40490FDB, 0xC0000000, 0x00000001])
    assert cast_c_frag(b, v, BF16, "generic") == b.vec_cast_f32_to(v, BF16)
    with pytest.raises(ValueError, match="bf16"):
        cast_c_frag(b, v, object(), "rne_noguard")
    with pytest.raises(ValueError):
        cast_c_frag(b, v, BF16, "rtz")
    with pytest.raises(ValueError, match="even"):
        rne_noguard_bf16(b, _f32_vec([0, 0, 0]))


# ---------------------------------------------------------------------------
# batched stats loads on host
# ---------------------------------------------------------------------------


def _stats_case(n_heads, ws_rows, *, pad):
    """Workspace arrays ``[h][rows]`` of distinct values (``pad`` extra floats
    after the array, as the plan's 256-byte rounding leaves)."""
    total = n_heads * ws_rows
    lse = np.arange(total + pad, dtype=np.float32) * 0.5 + 1.0
    dsum = -np.arange(total + pad, dtype=np.float32) * 0.25 - 3.0
    return lse, dsum


def _views(b, lse_ptr, dsum_ptr, ws_rows, row0):
    rows = b.const_i32(ws_rows)
    r0 = b.const_i64(row0)

    def ws(ptr):
        return WorkspaceAddr(b, ptr, ws_rows=rows, width=1, row0=r0)

    return ws(lse_ptr), ws(dsum_ptr)


def _eval_rows(n_heads, ws_rows, row0, h, q0, rows, width, pad):
    """``(scalar values, batched values, loads)`` of one step on host."""
    lse, dsum = _stats_case(n_heads, ws_rows, pad=pad)
    out = []
    for batched in (False, True):
        b = IntEvalBuilder({"lse": lse.copy(), "dsum": dsum.copy()})
        wl, wd = _views(b, b.ptr("lse"), b.ptr("dsum"), ws_rows, row0)
        hh, qq = b.const_i32(h), b.const_i32(q0)
        rr = [b.const_i32(r) for r in rows]
        if batched:
            vals = batched_stats_rows(
                b, wl, wd, hh, qq, rr, width=width, n_heads=b.const_i32(n_heads)
            )
        else:
            vals = StatsView(wl, wd).rows(b, hh, qq, rr)
        loads = [e for e in b.log if e[0] in ("load", "vload")]
        out.append(([(a.v, d.v) for a, d in vals], loads))
    return out


def _run_groups(width):
    # lane groups of a 16-row fragment (rows 4 * (lane // 16) + s) and a second
    # fragment row block (+16): every lane's runs
    base = [0, 4, 8, 12, 16, 20, 24, 28]
    return [r0 + j for r0 in base for j in range(4)], width


def _shapes():
    for n_heads, ws_rows in itertools.product((1, 2, 3), (1, 2, 3, 4, 5, 7, 9, 33)):
        for h in sorted({0, n_heads - 1}):
            for q0 in (0, 4, 16, 32, ws_rows - 3, ws_rows - 1, ws_rows, ws_rows + 5):
                if q0 < 0:
                    continue
                for row0 in (0, 1, 3):
                    yield n_heads, ws_rows, row0, h, q0


@pytest.mark.parametrize("width", STATS_VECS[1:])
def test_batched_stats_rows_are_the_scalar_rows(width):
    rows, _ = _run_groups(width)
    checked = 0
    for n_heads, ws_rows, row0, h, q0 in _shapes():
        total = n_heads * ws_rows
        if row0 + 1 > ws_rows and row0:
            continue  # the sequence starts past the workspace (plan declines)
        # the plan's rounding (A(4 * rows * h) bytes) when the whole array is
        # shorter than the window, else the exact array
        pad = (-(4 * total) % 256) // 4 if total < width else 0
        (sv, _), (bv, loads) = _eval_rows(
            n_heads, ws_rows, row0, h, q0, rows, width, pad
        )
        # identical values: in-range rows read their own element, rows past
        # ws_rows the neutral +inf / 0 (bit for bit, NaN-free inputs)
        assert sv == bv, (n_heads, ws_rows, row0, h, q0)
        assert all(e[0] == "vload" and e[2] == width for e in loads if e[0] != "load")
        assert len([e for e in loads if e[0] == "vload"]) == 2 * len(rows) // width
        checked += 1
    assert checked > 300


def test_batched_stats_window_needs_the_rounded_sub_buffer():
    """An array shorter than the window is read past its end (inside the
    plan's rounded sub-buffer); with an exact-size buffer the host evaluator
    reports the out-of-bounds access, so the contract is the plan's layout."""
    with pytest.raises(IndexError):
        _eval_rows(1, 2, 0, 0, 0, [0, 1, 2, 3], 4, pad=0)
    for h_q, rows_q in ((1, 1), (3, 1), (1, 3)):
        req = AttnBwdRequest(
            b=1, h_q=h_q, h_k=h_q, h_v=h_q, s_q=rows_q, s_kv=rows_q, d_qk=64, d_v=64
        )
        lay = attn_bwd_workspace_layout(req)
        end = attn_bwd_workspace_bytes(req)
        for name in ("WS_LSE2", "WS_DSUM"):
            off, nbytes = lay[name]
            assert nbytes == 4 * rows_q * h_q
            nxt = min([o for o, _ in lay.values() if o > off], default=end)
            assert nxt - off >= 4 * max(STATS_VECS), (name, nxt - off)


def test_batched_stats_rejects_other_layouts():
    b = IntEvalBuilder({"x": np.zeros(8, np.float32)})
    rows = b.const_i32(4)
    tok = WorkspaceAddr(
        b, b.ptr("x"), ws_rows=rows, width=1, row0=b.const_i64(0),
        layout="token_major", n_heads=2,
    )  # fmt: skip
    with pytest.raises(ValueError, match="head-major"):
        batched_stats_rows(
            b, tok, tok, b.const_i32(0), b.const_i32(0), [b.const_i32(0)] * 4,
            width=4, n_heads=b.const_i32(2),
        )  # fmt: skip


# ---------------------------------------------------------------------------
# IR: each lever changes exactly its ops
# ---------------------------------------------------------------------------

_ARCH_DTYPE = (("gfx942", "bf16"), ("gfx950", "bf16"), ("gfx942", "fp16"))


@cache
def _ir(arch, dtype, d, **kw):
    spec = AttnBwdSpec(
        head_size=d,
        dtype=dtype,
        mask_class="band",
        **config_knobs("default", arch),
        **kw,
    )
    name = validate_attn_bwd_spec(spec, arch).kernel_name("main")
    text = lower(build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22")
    return text.replace(name, "@K")


def _count(text, pat):
    return len(re.findall(pat, text))


_FPTRUNC_BF16 = r"= fptrunc <4 x float> %\S+ to <4 x bfloat>"
_PERM_HI = r"@llvm\.amdgcn\.perm\(i32 %\S+, i32 %\S+, i32 117834498\)"
_EXP2 = r"call float @llvm\.exp2\.f32\("
_EXP2_FAST = r"call float @llvm\.amdgcn\.exp2\.f32\("
_FMA = r"call float @llvm\.fmuladd\.f32\("
_LOAD_F32 = r"= load float, ptr addrspace\(1\)"


def test_explicit_defaults_emit_the_default_ir():
    plain = _ir("gfx942", "bf16", 64)
    assert plain == _ir(
        "gfx942", "bf16", 64, bf16_cast="generic", bwd_exp2="exact",
        stats_fma=False, stats_vec=1,
    )  # fmt: skip


@pytest.mark.parametrize("d", (64, 128))
def test_rne_noguard_replaces_the_p_ds_casts(d):
    plain = _ir("gfx942", "bf16", d)
    nog = _ir("gfx942", "bf16", d, bf16_cast="rne_noguard")
    removed = _count(plain, _FPTRUNC_BF16) - _count(nog, _FPTRUNC_BF16)
    assert removed > 0
    assert _count(plain, _PERM_HI) == 0
    assert _count(nog, _PERM_HI) == 2 * removed  # one perm per element pair
    # the vector adds of the rounding carry no nsw (a NaN word may wrap)
    assert re.search(r"= add <4 x i32>", nog)
    assert not re.search(r"= add nsw <4 x i32>", nog)


@pytest.mark.parametrize("arch,dtype", _ARCH_DTYPE)
def test_fast_exp2_replaces_every_p_exp2(arch, dtype):
    plain = _ir(arch, dtype, 64)
    fast = _ir(arch, dtype, 64, bwd_exp2="fast")
    n = _count(plain, _EXP2)
    assert n > 0 and _count(plain, _EXP2_FAST) == 0
    assert _count(fast, _EXP2) == 0 and _count(fast, _EXP2_FAST) == n


@pytest.mark.parametrize("arch,dtype", _ARCH_DTYPE)
def test_stats_fma_fuses_every_p_argument(arch, dtype):
    plain = _ir(arch, dtype, 64)
    fused = _ir(arch, dtype, 64, stats_fma=True)
    n = _count(plain, _EXP2)
    assert _count(fused, _FMA) - _count(plain, _FMA) == n
    assert _count(plain, r"= fsub float") - _count(fused, r"= fsub float") == n
    assert _count(fused, r"= fneg float") > 0


@pytest.mark.parametrize("width", STATS_VECS[1:])
@pytest.mark.parametrize("arch,dtype", _ARCH_DTYPE)
def test_stats_vec_batches_the_stats_loads(arch, dtype, width):
    plain = _ir(arch, dtype, 64)
    vec = _ir(arch, dtype, 64, stats_vec=width)
    pat = rf"= load <{width} x float>, ptr addrspace\(1\) %\S+, align 4"
    added = _count(vec, pat) - _count(plain, pat)
    assert added > 0 and added % 2 == 0  # one lse2 and one Dsum load per run
    assert _count(plain, _LOAD_F32) - _count(vec, _LOAD_F32) == width * added
