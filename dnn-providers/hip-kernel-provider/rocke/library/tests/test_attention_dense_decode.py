# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side checks of the dense-attention decode classes.

A GPU numerical test cannot tell two decodes apart: every decode is a bijection
onto the work space, so any of them computes the right answer. These tests run
each class's real ``grid`` and ``emit_decode`` on Python ints instead, so a
decode that silently covers the work space in a different order, drops or
duplicates items, or bakes NQB into a runtime-shape body is caught on the host.
"""

from __future__ import annotations

import itertools

import pytest

from kernels.common.attention_dense_decode import NONPERSIST_DECODES, PERSIST_DECODES
from kernels.gfx950.attention_dense import (
    GFX950_PERSIST_DECODES,
    Gfx950AttentionDenseSpec,
)


class _IntBuilder:
    """Evaluates the IR-builder integer ops the decodes emit on Python ints
    (the operands are non-negative, so floor division is the i32 division)."""

    const_i32 = staticmethod(int)
    add = staticmethod(lambda a, b: a + b)
    sub = staticmethod(lambda a, b: a - b)
    mul = staticmethod(lambda a, b: a * b)
    div = staticmethod(lambda a, b: a // b)
    mod = staticmethod(lambda a, b: a % b)
    cmp_lt = staticmethod(lambda a, b: a < b)
    cmp_ge = staticmethod(lambda a, b: a >= b)
    cmp_eq = staticmethod(lambda a, b: a == b)
    cmp_ne = staticmethod(lambda a, b: a != b)
    select = staticmethod(lambda c, a, b: a if c else b)


B_ = _IntBuilder()

# (Hq, Hkv, B, S): GQA, MHA, and a head count that is not a power of two.
_SHAPES = ((32, 8, 2, 2048), (64, 64, 1, 1024), (40, 8, 3, 512))


def _spec(Hq, Hkv, B, S, causal, **over):
    return Gfx950AttentionDenseSpec(
        batch=B, seqlen_q=S, seqlen_kv=S, num_query_heads=Hq, num_kv_heads=Hkv,
        head_size=128, causal=causal, dtype="fp16", **over)


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
    raise AssertionError(f"no expected order for nonpersist decode {name!r}")


def _decode_all(decode, spec, seqlen_q):
    gx, gy, gz = decode.grid(spec)
    out = {}
    for bz, by, bx in itertools.product(range(gz), range(gy), range(gx)):
        d = decode.emit_decode(B_, spec, bx, by, bz, seqlen_q)
        out[bx + gx * (by + gy * bz)] = (d.qb, d.hq, d.bt)
    return out


@pytest.mark.parametrize("name", sorted(NONPERSIST_DECODES))
@pytest.mark.parametrize("causal", (True, False))
@pytest.mark.parametrize("shape", _SHAPES)
def test_nonpersist_decode_linearizes_to_its_documented_order(name, causal, shape):
    decode = NONPERSIST_DECODES[name]
    spec = _spec(*shape, causal, nonpersist_decode=name)
    nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
    work = nqb * spec.num_query_heads * spec.batch
    # Baked (None) and runtime (the kernel's seqlen_q param) shapes must agree.
    for seqlen_q in (None, spec.seqlen_q):
        items = _decode_all(decode, spec, seqlen_q)
        assert sorted(items) == list(range(work))
        assert len(set(items.values())) == work, "decode is not a bijection"
        for wid, (qb, hq, bt) in items.items():
            assert _block_linear(spec, qb, hq, bt, nqb) == wid, (name, wid)


@pytest.mark.parametrize("name", sorted(NONPERSIST_DECODES))
def test_nonpersist_decode_reads_nqb_from_the_runtime_param(name):
    """Under a runtime shape one binary serves every seqlen: decoding with the
    param set to another seqlen must give that seqlen's mapping, not the spec's."""
    decode = NONPERSIST_DECODES[name]
    built = _spec(32, 8, 2, 2048, True, nonpersist_decode=name)
    launched = _spec(32, 8, 2, 4096, True, nonpersist_decode=name)
    assert _decode_all(decode, launched, launched.seqlen_q) == {
        wid: v for wid, v in _decode_all(decode, launched, None).items()}
    gx, gy, gz = decode.grid(launched)
    for bz, by, bx in itertools.product(range(gz), range(gy), range(gx)):
        got = decode.emit_decode(B_, built, bx, by, bz, launched.seqlen_q)
        want = decode.emit_decode(B_, launched, bx, by, bz, None)
        assert got == want


def _persist_specs():
    """Every persist decode on every shape where it is legal, with and without
    causal masking, plus the interleave variant of qb_major."""
    return list(_iter_persist_specs())


def _iter_persist_specs():
    for name in sorted(GFX950_PERSIST_DECODES - {"auto"}):
        for shape, causal in itertools.product(_SHAPES, (True, False)):
            Hq, Hkv, B, S = shape
            nqb = S // 256
            num_persistent = {
                "gqa_pair": nqb * Hkv * B,
                "gqa_pair_2phase": nqb * Hq * B // 2,
            }.get(name, 256)
            for interleave in ((False, True) if name == "qb_major" else (False,)):
                try:
                    spec = _spec(*shape, causal, persistent=True,
                                 num_persistent=num_persistent,
                                 persist_decode=name, interleave=interleave)
                except ValueError:
                    continue  # not legal for this shape (e.g. gqa_pair on MHA)
                yield pytest.param(spec, id=f"{name}-{Hq}x{Hkv}-B{B}-S{S}-"
                                   f"{'causal' if causal else 'full'}"
                                   f"{'-intl' if interleave else ''}")


@pytest.mark.parametrize("spec", _persist_specs())
def test_persist_decode_is_a_bijection(spec):
    decode = PERSIST_DECODES[spec.persist_decode]
    gqa = spec.num_queries_per_kv
    nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
    work = nqb * spec.num_query_heads * spec.batch
    items = set()
    for wi in range(work):
        d = decode.emit_decode(B_, spec, wi, None)
        assert 0 <= d.qb < nqb and 0 <= d.hq < spec.num_query_heads
        assert 0 <= d.bt < spec.batch
        assert d.hkv is None or d.hkv == d.hq // gqa
        items.add((d.qb, d.hq, d.bt))
    assert len(items) == work, "decode drops or duplicates work items"


def test_every_persist_decode_is_covered():
    covered = {p.values[0].persist_decode for p in _persist_specs()}
    assert covered == set(PERSIST_DECODES)
