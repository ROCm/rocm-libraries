# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the runtime-shape (AOT) coordinate transforms.

Covers:
  - the host-side magic-division contract: for every divisor in [1, 2**16] and
    a representative dividend set up to 2**31 - 1, the exact sequence
    do_magic_division_dynamic emits (umulhi, 32-bit add, lshr) reproduces
    x // d with the (multiplier, shift) from calculate_magic_numbers
  - do_magic_division_dynamic emits umul_hi -> add -> lshr on the runtime
    Values themselves (no constant substitution, lshr even for shift 0)
  - EmbedDynamic / UnmergeMagicDynamic / PadDynamic accept mixed int / Value
    arguments, multiply by runtime Values, and turn lo / hi into the validity
    predicate
  - DynamicTensorDescriptor.offset reduces with the runtime stride Values
  - the ValueError paths

Beyond structure, a small evaluator for the arith ops these helpers emit runs
the built IR on concrete inputs, so the descriptor chains are also checked
numerically against a plain-Python reference. CPU-only, numpy only.

The C++ twins are byte-compared by the dynamic_helpers parity family
(tests/instances/parity/).
"""

from __future__ import annotations

import numpy as np
import pytest

from rocke.core.ir import I32, IRBuilder, Value
from rocke.helpers.transforms import (
    CoordVar,
    DynamicTensorDescriptor,
    EmbedDynamic,
    PadDynamic,
    UnmergeMagicDynamic,
    calculate_magic_numbers,
    do_magic_division,
    do_magic_division_dynamic,
    embed_dynamic,
    pad_dynamic,
    unmerge_magic_dynamic,
)


MASK32 = (1 << 32) - 1
INT31_MAX = (1 << 31) - 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _builder():
    return IRBuilder("xform_dyn_test")


def _params(b, *names):
    return [b.param(n, I32) for n in names]


def _new_ops(b, start):
    return b.kernel.body.ops[start:]


def _to_signed(v):
    v &= MASK32
    return v - (1 << 32) if v >= (1 << 31) else v


def _evaluate(ops, env):
    """Run the i32 / i1 arith ops the dynamic transforms emit.

    ``env`` maps ``id(Value) -> int`` for the inputs and is extended in place
    with every result. i32 values are kept as unsigned 32-bit bit patterns,
    exactly as the hardware holds them; ``arith.cmp`` compares them signed,
    like the lowered ``icmp s*``.
    """
    for op in ops:
        n = op.name
        a = [env[id(v)] for v in op.operands]
        if n == "arith.constant":
            r = int(op.attrs["value"]) & MASK32
        elif n == "arith.add":
            r = (a[0] + a[1]) & MASK32
        elif n == "arith.sub":
            r = (a[0] - a[1]) & MASK32
        elif n == "arith.mul":
            r = (a[0] * a[1]) & MASK32
        elif n == "arith.umul_hi_i32":
            r = (a[0] * a[1]) >> 32
        elif n == "arith.lshr":
            r = a[0] >> a[1]
        elif n == "arith.and":
            r = a[0] & a[1]
        elif n == "arith.select":
            r = a[1] if a[0] else a[2]
        elif n == "arith.cmp":
            x, y = _to_signed(a[0]), _to_signed(a[1])
            r = int(
                {"lt": x < y, "ge": x >= y, "le": x <= y, "gt": x > y}[op.attrs["pred"]]
            )
        else:  # pragma: no cover - a new op kind means the helper changed
            raise AssertionError(f"evaluator does not model {n}")
        env[id(op.results[0])] = r
    return env


def _magic_reference(x, mult, shift):
    """Host model of do_magic_division_dynamic's emitted sequence."""
    tmp = (x * mult) >> 32  # arith.umul_hi_i32
    summed = (tmp + x) & MASK32  # arith.add on i32 (wraps)
    return summed >> shift  # arith.lshr


# ---------------------------------------------------------------------------
# Magic-division contract (host side)
# ---------------------------------------------------------------------------


class TestMagicNumbers:
    def test_small_divisors_exact(self):
        assert calculate_magic_numbers(1) == (1, 0)
        assert calculate_magic_numbers(2) == (1, 1)
        # Non power of two: shift = ceil(log2(d)).
        mult, shift = calculate_magic_numbers(3)
        assert shift == 2
        assert mult == ((4 - 3) << 32) // 3 + 1

    @pytest.mark.parametrize("d", [0, -1])
    def test_non_positive_divisor_rejected(self, d):
        with pytest.raises(ValueError, match="divisor >= 1"):
            calculate_magic_numbers(d)

    def test_multiplier_fits_uint32(self):
        # The multiplier travels as an i32 kernel arg; it must fit 32 bits.
        for d in list(range(1, 4097)) + [(1 << 16) - 1, 1 << 16]:
            mult, shift = calculate_magic_numbers(d)
            assert 0 < mult <= MASK32
            assert (1 << shift) >= d and (shift == 0 or (1 << (shift - 1)) < d)

    def test_contract_all_divisors_to_2_16(self):
        """((umulhi(x, mult) + x) >> shift) == x // d over the 31-bit range."""
        rng = np.random.default_rng(0x5EED)
        fixed = set()
        for p in range(31):
            for delta in (-1, 0, 1):
                v = (1 << p) + delta
                if 0 <= v <= INT31_MAX:
                    fixed.add(v)
        fixed.update(int(v) for v in rng.integers(0, INT31_MAX, size=48))
        fixed.add(INT31_MAX)
        fixed_arr = np.array(sorted(fixed), dtype=np.uint64)

        divisors = np.arange(1, (1 << 16) + 1, dtype=np.uint64)
        magics = [calculate_magic_numbers(int(d)) for d in divisors]
        mult = np.array([m for m, _ in magics], dtype=np.uint64)
        shift = np.array([s for _, s in magics], dtype=np.uint64)

        # Per-divisor boundary dividends: just below / at the largest multiple
        # of d in range, and around d itself.
        top = (INT31_MAX // divisors) * divisors
        per_d = np.stack(
            [top, top - 1, divisors - 1, divisors, divisors + 1, top + (divisors - 1)],
            axis=1,
        )
        per_d = np.minimum(per_d, INT31_MAX)

        mask = np.uint64(MASK32)
        for xs in (np.broadcast_to(fixed_arr, (divisors.size, fixed_arr.size)), per_d):
            m = mult[:, None]
            s = shift[:, None]
            # x < 2**31 and mult < 2**32, so x * mult < 2**63: exact in uint64.
            tmp = (xs * m) >> np.uint64(32)
            summed = (tmp + xs) & mask
            got = summed >> s
            want = xs // divisors[:, None]
            bad = np.argwhere(got != want)
            assert bad.size == 0, (
                f"magic division mismatch at d={int(divisors[bad[0][0]])}, "
                f"x={int(xs[bad[0][0], bad[0][1]])}"
            )

    def test_add_does_not_wrap_in_contract_range(self):
        # tmp < x whenever mult < 2**32, so tmp + x < 2**32 for x < 2**31:
        # the i32 add in the emitted sequence never wraps inside the contract.
        for d in (3, 7, 641, 65521, 1 << 16):
            mult, shift = calculate_magic_numbers(d)
            x = INT31_MAX
            tmp = (x * mult) >> 32
            assert tmp + x <= MASK32
            assert _magic_reference(x, mult, shift) == x // d

    def test_reference_matches_static_emitter_constants(self):
        """do_magic_division bakes the same multiplier bit pattern as a const.

        d = 5 has a multiplier >= 2**31, so the constant is stored as its
        negative two's-complement i32; the unsigned umul_hi must still see the
        original bits.
        """
        d = 5
        mult, shift = calculate_magic_numbers(d)
        assert mult >= 1 << 31
        b = _builder()
        (x,) = _params(b, "x")
        start = len(b.kernel.body.ops)
        q = do_magic_division(b, x, mult, shift)
        env = _evaluate(_new_ops(b, start), {id(x): 1_234_567})
        assert env[id(q)] == 1_234_567 // d


# ---------------------------------------------------------------------------
# do_magic_division_dynamic (IR structure)
# ---------------------------------------------------------------------------


class TestMagicDivisionDynamic:
    def test_emits_umulhi_add_lshr_on_runtime_values(self):
        b = _builder()
        x, mult, shift = _params(b, "x", "mult", "shift")
        start = len(b.kernel.body.ops)
        q = do_magic_division_dynamic(b, x, mult, shift)
        ops = _new_ops(b, start)
        assert [op.name for op in ops] == [
            "arith.umul_hi_i32",
            "arith.add",
            "arith.lshr",
        ]
        umh, add, lshr = ops
        assert umh.operands[0] is x and umh.operands[1] is mult
        assert add.operands[0] is umh.results[0] and add.operands[1] is x
        assert lshr.operands[0] is add.results[0] and lshr.operands[1] is shift
        assert q is lshr.results[0]

    def test_no_constants_materialized(self):
        b = _builder()
        x, mult, shift = _params(b, "x", "mult", "shift")
        start = len(b.kernel.body.ops)
        do_magic_division_dynamic(b, x, mult, shift)
        assert not any(op.name == "arith.constant" for op in _new_ops(b, start))

    def test_lshr_kept_even_for_zero_shift(self):
        # The static variant drops the shift for shift == 0; the dynamic one
        # cannot know the shift, so it always shifts.
        b = _builder()
        (x,) = _params(b, "x")
        start = len(b.kernel.body.ops)
        do_magic_division(b, x, 1, 0)
        assert "arith.lshr" not in [op.name for op in _new_ops(b, start)]

        b = _builder()
        x, mult, shift = _params(b, "x", "mult", "shift")
        start = len(b.kernel.body.ops)
        do_magic_division_dynamic(b, x, mult, shift)
        assert _new_ops(b, start)[-1].name == "arith.lshr"

    @pytest.mark.parametrize("d", [1, 2, 3, 7, 12, 255, 1000, 65535, 65536])
    def test_evaluates_to_quotient(self, d):
        mult, shift = calculate_magic_numbers(d)
        b = _builder()
        x, m, s = _params(b, "x", "mult", "shift")
        start = len(b.kernel.body.ops)
        q = do_magic_division_dynamic(b, x, m, s)
        for xv in (0, 1, d - 1, d, d + 1, 12345678, INT31_MAX):
            env = {id(x): xv, id(m): mult & MASK32, id(s): shift}
            _evaluate(_new_ops(b, start), env)
            assert env[id(q)] == xv // d


# ---------------------------------------------------------------------------
# EmbedDynamic
# ---------------------------------------------------------------------------


class TestEmbedDynamic:
    def _apply(self, t, b, **coords):
        cvs = {
            k: (v if isinstance(v, CoordVar) else CoordVar(k, v))
            for k, v in coords.items()
        }
        start = len(b.kernel.body.ops)
        out = t.apply(b, cvs)
        return out, _new_ops(b, start)

    def test_value_stride_used_directly(self):
        b = _builder()
        u, s = _params(b, "u", "s")
        t = embed_dynamic(("u",), "l", strides=[s])
        out, ops = self._apply(t, b, u=u)
        muls = [op for op in ops if op.name == "arith.mul"]
        assert len(muls) == 1
        assert muls[0].operands == [u, s]
        # offset=0 is an int: its constant is built but never added.
        assert [op.name for op in ops] == ["arith.mul", "arith.constant"]
        assert out["l"].value is muls[0].results[0]
        assert out["l"].valid is None

    def test_mixed_int_and_value_strides(self):
        b = _builder()
        u0, u1, u2, s0 = _params(b, "u0", "u1", "u2", "s0")
        t = EmbedDynamic(("u0", "u1", "u2"), "l", strides=[s0, 1, 3], offset=-2)
        out, ops = self._apply(t, b, u0=u0, u1=u1, u2=u2)
        muls = [op for op in ops if op.name == "arith.mul"]
        # Value stride and int stride 3 multiply; int stride 1 does not.
        assert len(muls) == 2
        assert muls[0].operands == [u0, s0]
        assert muls[1].operands[0] is u2
        assert muls[1].operands[1].op.attrs["value"] == 3
        assert not any(u1 is op.operands[0] for op in muls)
        consts = [op.attrs["value"] for op in ops if op.name == "arith.constant"]
        # The dead stride-1 constant is still emitted (the C++ twin mirrors
        # it to keep SSA numbering aligned), then 3, then the offset.
        assert consts == [1, 3, -2]
        env = {id(u0): 5, id(u1): 7, id(u2): 11, id(s0): 13}
        _evaluate(ops, env)
        assert _to_signed(env[id(out["l"].value)]) == 5 * 13 + 7 + 11 * 3 - 2

    def test_value_offset_is_added(self):
        b = _builder()
        u, off = _params(b, "u", "off")
        t = embed_dynamic(("u",), "l", strides=[1], offset=off)
        out, ops = self._apply(t, b, u=u)
        adds = [op for op in ops if op.name == "arith.add"]
        assert len(adds) == 1 and adds[0].operands == [u, off]
        assert out["l"].value is adds[0].results[0]

    def test_lo_hi_values_form_valid_predicate(self):
        b = _builder()
        u, s, lo, hi = _params(b, "u", "s", "lo", "hi")
        t = embed_dynamic(("u",), "l", strides=[s], lo=lo, hi=hi)
        out, ops = self._apply(t, b, u=u)
        acc = out["l"].value
        cmps = [op for op in ops if op.name == "arith.cmp"]
        assert [(c.attrs["pred"], c.operands[0], c.operands[1]) for c in cmps] == [
            ("ge", acc, lo),
            ("lt", acc, hi),
        ]
        ands = [op for op in ops if op.name == "arith.and"]
        assert len(ands) == 1
        assert ands[0].operands == [cmps[0].results[0], cmps[1].results[0]]
        assert out["l"].valid is ands[0].results[0]
        for uv, want in ((0, 0), (3, 1), (9, 0)):
            env = {id(u): uv, id(s): 2, id(lo): 2, id(hi): 17}
            _evaluate(ops, env)
            assert env[id(out["l"].valid)] == want

    def test_int_lo_value_hi(self):
        b = _builder()
        u, hi = _params(b, "u", "hi")
        t = embed_dynamic(("u",), "l", strides=[1], offset=-1, lo=0, hi=hi)
        _, ops = self._apply(t, b, u=u)
        cmps = [op for op in ops if op.name == "arith.cmp"]
        assert cmps[0].attrs["pred"] == "ge"
        assert cmps[0].operands[1].op.attrs["value"] == 0
        assert cmps[1].attrs["pred"] == "lt" and cmps[1].operands[1] is hi

    def test_incoming_validity_is_conjoined(self):
        b = _builder()
        u, hi, flag = _params(b, "u", "hi", "flag")
        in_valid = b.cmp_lt(flag, hi)
        t = embed_dynamic(("u",), "l", strides=[1], hi=hi)
        out, ops = self._apply(t, b, u=CoordVar("u", u, in_valid))
        last = ops[-1]
        assert last.name == "arith.and"
        assert last.operands[0] is in_valid
        assert out["l"].valid is last.results[0]

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="len\\(upper\\) == len\\(strides\\)"):
            EmbedDynamic(("a", "b"), "l", strides=[1])


# ---------------------------------------------------------------------------
# UnmergeMagicDynamic
# ---------------------------------------------------------------------------


class TestUnmergeMagicDynamic:
    def _build(self, dims_as_values=True):
        b = _builder()
        m = b.param("m", I32)
        names = ("m1", "s1", "d1", "m2", "s2", "d2")
        m1, s1, d1, m2, s2, d2 = _params(b, *names)
        t = unmerge_magic_dynamic("m", ("n", "h", "w"), [(m1, s1, d1), (m2, s2, d2)])
        start = len(b.kernel.body.ops)
        out = t.apply(b, {"m": CoordVar("m", m)})
        return b, m, (m1, s1, d1, m2, s2, d2), out, _new_ops(b, start)

    def test_divides_by_runtime_triples_last_first(self):
        _, m, (m1, s1, _, m2, s2, _), out, ops = self._build()
        umh = [op for op in ops if op.name == "arith.umul_hi_i32"]
        lshr = [op for op in ops if op.name == "arith.lshr"]
        assert len(umh) == 2 and len(lshr) == 2
        # The last lower ("w") is peeled first with the last triple.
        assert umh[0].operands == [m, m2] and lshr[0].operands[1] is s2
        q_w = lshr[0].results[0]
        assert umh[1].operands == [q_w, m1] and lshr[1].operands[1] is s1
        assert out["n"].value is lshr[1].results[0]
        assert not any(op.name == "arith.constant" for op in ops)

    def test_remainders_use_runtime_dims(self):
        _, m, (_, _, d1, _, _, d2), out, ops = self._build()
        muls = [op for op in ops if op.name == "arith.mul"]
        assert [op.operands[1] for op in muls] == [d2, d1]
        subs = [op for op in ops if op.name == "arith.sub"]
        assert out["w"].value is subs[0].results[0]
        assert out["h"].value is subs[1].results[0]
        assert subs[0].operands[0] is m

    @pytest.mark.parametrize("dims", [(3, 5), (7, 1), (64, 56), (255, 17)])
    def test_evaluates_to_div_mod(self, dims):
        _, m, (m1, s1, d1, m2, s2, d2), out, ops = self._build()
        dh, dw = dims
        mh, sh = calculate_magic_numbers(dh)
        mw, sw = calculate_magic_numbers(dw)
        for mv in (0, 1, dw, dh * dw - 1, dh * dw, 123457, 9 * dh * dw + 5):
            env = {
                id(m): mv,
                id(m1): mh,
                id(s1): sh,
                id(d1): dh,
                id(m2): mw,
                id(s2): sw,
                id(d2): dw,
            }
            _evaluate(ops, env)
            assert env[id(out["w"].value)] == mv % dw
            assert env[id(out["h"].value)] == (mv // dw) % dh
            assert env[id(out["n"].value)] == mv // (dw * dh)

    def test_int_dim_one_skips_division(self):
        b = _builder()
        m = b.param("m", I32)
        t = UnmergeMagicDynamic("m", ("a", "b"), [(1, 0, 1)])
        start = len(b.kernel.body.ops)
        out = t.apply(b, {"m": CoordVar("m", m)})
        ops = _new_ops(b, start)
        assert "arith.umul_hi_i32" not in [op.name for op in ops]
        assert out["a"].value is m
        assert out["b"].value.op.attrs["value"] == 0

    def test_validity_propagates_to_every_lower(self):
        b = _builder()
        m, hi, mu, sh, dm = _params(b, "m", "hi", "mu", "sh", "dm")
        in_valid = b.cmp_lt(m, hi)
        t = unmerge_magic_dynamic("m", ("a", "b"), [(mu, sh, dm)])
        out = t.apply(b, {"m": CoordVar("m", m, in_valid)})
        assert out["a"].valid is in_valid and out["b"].valid is in_valid

    def test_triple_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="len\\(lowers\\)-1 magic triples"):
            UnmergeMagicDynamic("m", ("a", "b", "c"), [(1, 0, 1)])


# ---------------------------------------------------------------------------
# PadDynamic
# ---------------------------------------------------------------------------


class TestPadDynamic:
    def test_value_passes_through_hi_only(self):
        b = _builder()
        u, hi = _params(b, "u", "hi")
        start = len(b.kernel.body.ops)
        out = pad_dynamic("u", hi=hi).apply(b, {"u": CoordVar("u", u)})
        ops = _new_ops(b, start)
        assert [op.name for op in ops] == ["arith.cmp"]
        assert ops[0].attrs["pred"] == "lt" and ops[0].operands == [u, hi]
        assert out["u"].value is u
        assert out["u"].valid is ops[0].results[0]

    def test_int_lo_value_hi(self):
        b = _builder()
        u, hi = _params(b, "u", "hi")
        start = len(b.kernel.body.ops)
        out = PadDynamic("u", lo=2, hi=hi).apply(b, {"u": CoordVar("u", u)})
        ops = _new_ops(b, start)
        assert [op.name for op in ops] == [
            "arith.constant",
            "arith.cmp",
            "arith.cmp",
            "arith.and",
        ]
        for uv, want in ((1, 0), (2, 1), (9, 1), (10, 0)):
            env = {id(u): uv, id(hi): 10}
            _evaluate(ops, env)
            assert env[id(out["u"].valid)] == want

    def test_no_bounds_no_predicate(self):
        b = _builder()
        (u,) = _params(b, "u")
        start = len(b.kernel.body.ops)
        out = pad_dynamic("u").apply(b, {"u": CoordVar("u", u)})
        assert _new_ops(b, start) == []
        assert out["u"].valid is None


# ---------------------------------------------------------------------------
# DynamicTensorDescriptor
# ---------------------------------------------------------------------------


class TestDynamicTensorDescriptor:
    def test_offset_multiplies_by_runtime_strides(self):
        b = _builder()
        s_h, s_w, h, w = _params(b, "s_h", "s_w", "h", "w")
        one = b.const_i32(1)
        desc = DynamicTensorDescriptor.create(
            "t", coord_names=("h", "w", "c"), strides=(s_h, s_w, one)
        )
        (c,) = _params(b, "c")
        start = len(b.kernel.body.ops)
        off, valid = desc.offset(b, h=h, w=w, c=c)
        ops = _new_ops(b, start)
        muls = [op for op in ops if op.name == "arith.mul"]
        # Every base coord multiplies -- even by a stride that is a constant 1
        # Value -- and no stride constant is materialized.
        assert [op.operands for op in muls] == [[h, s_h], [w, s_w], [c, one]]
        assert not any(op.name == "arith.constant" for op in ops)
        assert valid is None
        env = {id(s_h): 96, id(s_w): 3, id(h): 4, id(w): 5, id(c): 2, id(one): 1}
        _evaluate(ops, env)
        assert env[id(off)] == 4 * 96 + 5 * 3 + 2

    def test_create_records_strides_and_names(self):
        b = _builder()
        s0, s1 = _params(b, "s0", "s1")
        desc = DynamicTensorDescriptor.create(
            "t", coord_names=("a", "b"), strides=(s0, s1)
        )
        assert desc.base_names == ("a", "b")
        assert desc.upper_names == ("a", "b")
        assert desc.dynamic_strides == (s0, s1)
        # Transforms keep the runtime strides (and the dynamic type).
        t = desc.transform(embed_dynamic(("x",), "a", strides=[s0]))
        assert isinstance(t, DynamicTensorDescriptor)
        assert t.dynamic_strides == (s0, s1)
        assert t.upper_names == ("b", "x")

    def test_chain_evaluates_like_reference(self):
        """unmerge_magic_dynamic + embed_dynamic chain vs a Python reference.

        m -> (ho, wo) by magic division; h = ho*cs + dy - 1 in [0, H);
        w = wo*2 in [0, W); offset = h*s_h + w*s_w.
        """
        b = _builder()
        names = ("H", "W", "s_h", "s_w", "cs", "mu", "sh", "Wo", "m", "dy")
        H, W, s_h, s_w, cs, mu, sh, wo_dim, m, dy = _params(b, *names)
        desc = DynamicTensorDescriptor.create(
            "x", coord_names=("h", "w"), strides=(s_h, s_w)
        ).transform(
            unmerge_magic_dynamic("m", ("ho", "wo"), [(mu, sh, wo_dim)]),
            embed_dynamic(("ho", "dy"), "h", strides=[cs, 1], offset=-1, lo=0, hi=H),
            embed_dynamic(("wo",), "w", strides=[2], offset=0, lo=0, hi=W),
        )
        assert set(desc.upper_names) == {"m", "dy"}
        start = len(b.kernel.body.ops)
        off, valid = desc.offset(b, m=m, dy=dy)
        ops = _new_ops(b, start)
        assert valid is not None

        Hv, Wv, Wo, csv = 9, 14, 7, 2
        mult, shift = calculate_magic_numbers(Wo)
        for mv in range(0, 5 * Wo, 3):
            for dyv in (0, 1, 2):
                env = {
                    id(H): Hv,
                    id(W): Wv,
                    id(s_h): Wv * 8,
                    id(s_w): 8,
                    id(cs): csv,
                    id(mu): mult,
                    id(sh): shift,
                    id(wo_dim): Wo,
                    id(m): mv,
                    id(dy): dyv,
                }
                _evaluate(ops, env)
                ho, wo = divmod(mv, Wo)
                h = ho * csv + dyv - 1
                w = wo * 2
                assert _to_signed(env[id(off)]) == h * Wv * 8 + w * 8
                assert env[id(valid)] == int(0 <= h < Hv and 0 <= w < Wv)

    def test_missing_upper_coord_rejected(self):
        b = _builder()
        s0, s1, a = _params(b, "s0", "s1", "a")
        desc = DynamicTensorDescriptor.create(
            "t", coord_names=("a", "b"), strides=(s0, s1)
        )
        with pytest.raises(ValueError, match="missing upper coords"):
            desc.offset(b, a=a)

    def test_missing_chain_input_rejected(self):
        b = _builder()
        s0, s1, x = _params(b, "s0", "s1", "x")
        desc = DynamicTensorDescriptor.create(
            "t", coord_names=("a", "b"), strides=(s0, s1)
        ).transform(embed_dynamic(("x",), "a", strides=[s0]))
        with pytest.raises(ValueError, match="missing upper coords"):
            desc.offset(b, x=x)

    def test_create_length_mismatch_rejected(self):
        b = _builder()
        (s0,) = _params(b, "s0")
        with pytest.raises(ValueError, match="length mismatch"):
            DynamicTensorDescriptor.create("t", coord_names=("a", "b"), strides=(s0,))

    def test_strides_are_values(self):
        b = _builder()
        s0, s1 = _params(b, "s0", "s1")
        desc = DynamicTensorDescriptor.create(
            "t", coord_names=("a", "b"), strides=(s0, s1)
        )
        assert all(isinstance(s, Value) for s in desc.dynamic_strides)
