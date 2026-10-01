# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side checks of the dense-attention decode classes.

A GPU numerical test cannot tell two decodes apart: every decode is a bijection
onto the work space, so any of them computes the right answer. These tests run
each class's real ``grid`` and ``emit_decode`` on Python ints instead, so a
decode that silently covers the work space in a different order, drops or
duplicates items, or bakes a shape value into a runtime-shape body is caught on
the host.

Runtime shapes are checked the way a runtime-shape binary runs: the decode is
emitted for one spec (the compile-time fields) and fed another launch's shape
(and, on the persistent grid, that launch's fast-division kernargs); it must give
the launched spec's baked mapping. The fake builder's comparisons return
``numpy.bool_``, not ``bool``, so a runtime flag takes the ``select`` path.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

from kernels.common.attention_dense_decode import (
    NONPERSIST_DECODES,
    PERSIST_DECODES,
    DecodeShape,
)
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    attention_dense_runtime_args,
)
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
)


def _sdiv(a, b):
    """i32 ``sdiv``: truncates toward zero (an unselected path may divide a
    negative value; Python's ``//`` would floor it)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class _IntBuilder:
    """Evaluates the IR-builder integer ops the decodes emit on Python ints, with
    i32 semantics. Comparisons return ``numpy.bool_`` so a runtime flag is not a
    ``bool`` and :meth:`DecodeShape.branch` takes its ``select`` path."""

    const_i32 = staticmethod(int)
    add = staticmethod(lambda a, b: a + b)
    sub = staticmethod(lambda a, b: a - b)
    mul = staticmethod(lambda a, b: a * b)
    div = staticmethod(_sdiv)
    mod = staticmethod(lambda a, b: a - _sdiv(a, b) * b)
    lshr = staticmethod(lambda a, b: (a & 0xFFFFFFFF) >> b)
    umul_hi_i32 = staticmethod(lambda a, b: ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) >> 32)
    smax = staticmethod(max)
    land = staticmethod(lambda a, b: a & b)
    cmp_lt = staticmethod(lambda a, b: np.bool_(a < b))
    cmp_ge = staticmethod(lambda a, b: np.bool_(a >= b))
    cmp_eq = staticmethod(lambda a, b: np.bool_(a == b))
    cmp_ne = staticmethod(lambda a, b: np.bool_(a != b))
    select = staticmethod(lambda c, a, b: a if c else b)


B_ = _IntBuilder()

_SHAPE_FIELDS = ("batch", "seqlen_q", "seqlen_kv", "num_query_heads", "num_kv_heads")


def _baked(spec):
    return DecodeShape(B_, spec)


def _runtime(built, launched, fastdiv=None, extra=None):
    """``built``'s decode fed ``launched``'s shape, as one runtime-shape binary
    serves another launch."""
    values = {f: getattr(launched, f) for f in _SHAPE_FIELDS}
    values["gqa"] = launched.num_queries_per_kv
    values.update(extra or {})
    return DecodeShape(B_, built, values, fastdiv=fastdiv)


def _fastdiv(launched, dividends):
    """The kernel's fast division, from the launch's real host kernargs."""
    args = attention_dense_runtime_args(launched)

    def div(n, d):
        dividends.append(n)
        q = (B_.umul_hi_i32(n, args[f"{d}_magic"]) + n) >> args[f"{d}_shift"]
        return q

    return div


# (Hq, Hkv, B, S): GQA, MHA, and a head count that is not a power of two.
_SHAPES = ((32, 8, 2, 2048), (64, 64, 1, 1024), (40, 8, 3, 512))


def _spec(Hq, Hkv, B, S, causal, **over):
    return Gfx950AttentionDenseSpec(
        batch=B,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=128,
        causal=causal,
        dtype="fp16",
        **over,
    )


def _block_linear(spec, qb, hq, bt, nqb):
    """Expected hardware dispatch index of the CTA for (qb, hq, bt), per decode.
    Written from each order's definition, independently of the decode code;
    ``blk`` is the query block before any causal reversal."""
    Hq, Hkv, B = spec.num_query_heads, spec.num_kv_heads, spec.batch
    gqa = Hq // Hkv
    name = spec.nonpersist_decode
    if name == "qb_minor":  # digits, fastest first: Q, head, B
        return qb + nqb * (hq + Hq * bt)
    if name == "bt_hkv_minor":  # B, V, G, Q; Q longest-first under causal
        blk = nqb - 1 - qb if spec.causal else qb
        hkv, hql = divmod(hq, gqa)
        return bt + B * (hkv + Hkv * (hql + gqa * blk))
    if name == "hq_minor_swz":
        # arXiv 2511.02132 Fig. 11 per batch, with batch the slowest digit:
        #   head = (w % M)*(Hq/M) + w // (M*nqb),  block = (w % (M*nqb)) // M
        # inverted to w for (hq, blk); query blocks longest-first under causal.
        M = spec.chiplet_num_xcds
        blk = nqb - 1 - qb if spec.causal else qb
        a, c = divmod(hq, Hq // M)
        return a + M * blk + M * nqb * c + Hq * nqb * bt
    raise AssertionError(f"no expected order for nonpersist decode {name!r}")


def _decode_all(decode, spec, make_shape):
    gx, gy, gz = decode.grid(spec)
    out = {}
    for bz, by, bx in itertools.product(range(gz), range(gy), range(gx)):
        d = decode.emit_decode(B_, spec, bx, by, bz, make_shape(spec))
        out[bx + gx * (by + gy * bz)] = (d.qb, d.hq, d.bt)
    return out


# Baked; the gfx950 grid (only seqlen_q is a param); the gfx942 runtime grid.
_GRID_SHAPES = {
    "baked": _baked,
    "seqlen_q": lambda s: DecodeShape(B_, s, {"seqlen_q": s.seqlen_q}),
    "runtime": lambda s: _runtime(s, s),
}


@pytest.mark.parametrize("name", sorted(NONPERSIST_DECODES))
@pytest.mark.parametrize("causal", (True, False))
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("xcds", (8, 4))
def test_nonpersist_decode_linearizes_to_its_documented_order(
    name, causal, shape, xcds
):
    decode = NONPERSIST_DECODES[name]
    spec = _spec(*shape, causal, nonpersist_decode=name, chiplet_num_xcds=xcds)
    nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
    work = nqb * spec.num_query_heads * spec.batch
    for make_shape in _GRID_SHAPES.values():
        items = _decode_all(decode, spec, make_shape)
        assert sorted(items) == list(range(work))
        assert len(set(items.values())) == work, "decode is not a bijection"
        for wid, (qb, hq, bt) in items.items():
            assert _block_linear(spec, qb, hq, bt, nqb) == wid, (name, wid)


# Launches one runtime-shape grid binary serves: other batch, seqlen and heads.
_GRID_LAUNCHES = (
    (32, 8, 2, 4096),
    (64, 64, 1, 1024),
    (40, 8, 3, 512),
    (16, 4, 2, 1024),
)


@pytest.mark.parametrize("name", sorted(NONPERSIST_DECODES))
def test_nonpersist_decode_reads_the_shape_from_the_runtime_params(name):
    """Under a runtime shape one binary serves every shape: the decode built for
    one spec and fed another launch's shape must give that launch's baked
    mapping, not the built spec's."""
    decode = NONPERSIST_DECODES[name]
    built = _spec(32, 8, 2, 2048, True, nonpersist_decode=name)
    for Hq, Hkv, B, S in _GRID_LAUNCHES:
        launched = dataclasses.replace(
            built,
            batch=B,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
        )
        gx, gy, gz = decode.grid(launched)
        for bz, by, bx in itertools.product(range(gz), range(gy), range(gx)):
            got = decode.emit_decode(B_, built, bx, by, bz, _runtime(built, launched))
            want = decode.emit_decode(B_, launched, bx, by, bz, _baked(launched))
            assert got == want, (name, (Hq, Hkv, B, S), (bx, by, bz))


def _persist_specs():
    """Every persist decode on every shape where it is legal, with and without
    causal masking, plus the interleave variant of qb_major."""
    return list(_iter_persist_specs())


def _iter_persist_specs():
    for name in sorted(Gfx950AttentionDenseSpec.supported_persist_decodes() - {"auto"}):
        for shape, causal in itertools.product(_SHAPES, (True, False)):
            Hq, Hkv, B, S = shape
            nqb = S // 256
            num_persistent = {
                "gqa_pair": nqb * Hkv * B,
                "gqa_pair_2phase": nqb * Hq * B // 2,
            }.get(name, 256)
            for interleave in ((False, True) if name == "qb_major" else (False,)):
                try:
                    spec = _spec(
                        *shape,
                        causal,
                        persistent=True,
                        num_persistent=num_persistent,
                        persist_decode=name,
                        interleave=interleave,
                    )
                except ValueError:
                    continue  # not legal for this shape (e.g. gqa_pair on MHA)
                yield pytest.param(
                    spec,
                    id=f"{name}-{Hq}x{Hkv}-B{B}-S{S}-"
                    f"{'causal' if causal else 'full'}"
                    f"{'-intl' if interleave else ''}",
                )


@pytest.mark.parametrize("spec", _persist_specs())
def test_persist_decode_is_a_bijection(spec):
    decode = PERSIST_DECODES[spec.persist_decode]
    gqa = spec.num_queries_per_kv
    nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
    work = nqb * spec.num_query_heads * spec.batch
    items = set()
    for wi in range(work):
        d = decode.emit_decode(B_, spec, wi, _baked(spec))
        assert 0 <= d.qb < nqb and 0 <= d.hq < spec.num_query_heads
        assert 0 <= d.bt < spec.batch
        assert d.hkv is None or d.hkv == d.hq // gqa
        items.add((d.qb, d.hq, d.bt))
    assert len(items) == work, "decode drops or duplicates work items"


def test_every_persist_decode_is_covered():
    covered = {p.values[0].persist_decode for p in _persist_specs()}
    assert covered == set(PERSIST_DECODES)


def test_bt_hkv_minor_follows_its_documented_order():
    """``wi = ((blk*gqa + hql)*Hkv + hkv)*B + bt`` (the class docstring), checked
    independently of the decode's Hq-first split of the head digits."""
    for Hq, Hkv, B, S in _SHAPES + ((48, 6, 2, 1024),):
        spec = _spec(
            Hq,
            Hkv,
            B,
            S,
            False,
            persistent=True,
            num_persistent=256,
            persist_decode="bt_hkv_minor",
        )
        gqa, nqb = Hq // Hkv, S // 256
        for wi in range(nqb * Hq * B):
            d = PERSIST_DECODES["bt_hkv_minor"].emit_decode(B_, spec, wi, _baked(spec))
            hkv, hql = d.hq // gqa, d.hq % gqa
            assert d.hkv == hkv
            assert ((d.qb * gqa + hql) * Hkv + hkv) * B + d.bt == wi


def _gfx942_persist(Hq, Hkv, B, S, name, causal=True, window=0, np_=256, **over):
    return Gfx942AttentionDenseSpec(
        batch=B,
        seqlen_q=S,
        seqlen_kv=S,
        num_query_heads=Hq,
        num_kv_heads=Hkv,
        head_size=128,
        causal=causal,
        dtype="fp16",
        sliding_window=window,
        persistent=True,
        num_persistent=np_,
        persist_decode=name,
        **over,
    )


# (Hq, Hkv, B, S) launches of one persistent binary: NQB even and odd, W below,
# near and far above NP, GQA and MHA, a head count that is not a power of two.
_PERSIST_LAUNCHES = (
    (32, 8, 1, 512),
    (32, 8, 2, 2048),
    (40, 8, 1, 768),
    (64, 64, 1, 1024),
    (16, 4, 3, 1280),
    (48, 8, 4, 4096),
)


@pytest.mark.parametrize(
    "name", sorted(Gfx942AttentionDenseSpec.supported_persist_decodes() - {"auto"})
)
@pytest.mark.parametrize("causal,window", ((True, 0), (False, 0)))
@pytest.mark.parametrize("interleave", (False, True))
def test_runtime_persist_decode_matches_the_launched_baked_decode(
    name, causal, window, interleave
):
    """One gfx942 persistent binary (built for the first launch) serves every
    launch through the shape params and that launch's fast-division kernargs: it
    must decode exactly as the launch's own baked decode, every fast-division
    dividend must stay within the decode's bound, and the work must be covered
    once."""
    if interleave and name != "qb_major":
        pytest.skip("interleave applies to qb_major only")
    decode = PERSIST_DECODES[name]
    built = None
    for Hq, Hkv, B, S in _PERSIST_LAUNCHES:
        try:
            launched = _gfx942_persist(
                Hq, Hkv, B, S, name, causal, window, interleave=interleave
            )
        except ValueError:
            continue  # not legal for this launch (e.g. hq_minor_swz, odd NQB)
        built = built or launched
        nqb = S // launched.block_m
        work = nqb * Hq * B
        dividends = []
        shape = _runtime(
            built, launched, _fastdiv(launched, dividends), {"nqb": nqb, "w": work}
        )
        got = [decode.emit_decode(B_, built, wi, shape) for wi in range(work)]
        want = [
            decode.emit_decode(B_, launched, wi, _baked(launched)) for wi in range(work)
        ]
        assert got == want, (name, (Hq, Hkv, B, S))
        assert len({(d.qb, d.hq, d.bt) for d in got}) == work
        assert max(dividends, default=0) <= decode.fastdiv_dividend_bound(launched)
    assert built is not None, f"no legal launch for {name}"
