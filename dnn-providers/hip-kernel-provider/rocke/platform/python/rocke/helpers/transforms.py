# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CK Tile-style coordinate-transform DAG over SSA values.

CK Tile's central idea is to compose convolution / attention / GEMM
addressing as a *coordinate-transform DAG* on top of a naive
multi-dimensional tensor descriptor. This module provides that
algebra at the level of the Python IR: every "coordinate" is an
i32 SSA `Value` with an optional i1 "is in bounds" predicate, and
every "transform" is a pure Python function that the kernel builder
calls to extend the DAG.

It is the same set of building blocks `_descriptor.py` exposes for
C++ codegen (`pass_through`, `pad`, `merge`, `unmerge`,
`embed`-as-affine, `transform_tensor_descriptor`) but operating on
runtime SSA values instead of emitting C++ template instantiations.
The kernel author writes:

    # naive descriptor (input image in NHWC layout)
    in_desc = NaiveTensorDescriptor.from_shape("A", [N, Hi, Wi, C], dtype=F16)
    # transform: pad H and W with PAD on either side (validity only)
    in_desc = in_desc.transform(
        pad("h", lo=0, hi=Hi),
        pad("w", lo=0, hi=Wi),
    )
    # transform: embed (ho, r) -> hi via stride=sH offset=-pH dilation=dH
    in_desc = in_desc.transform(
        embed(("ho", "r"), "h", strides=(sH, dH), offset=-pH),
        embed(("wo", "s"), "w", strides=(sW, dW), offset=-pW),
    )
    # transform: package (n, ho, wo) as the implicit-GEMM M dimension
    in_desc = in_desc.transform(merge(("n", "ho", "wo"), into="m"))
    # transform: package (r, s, c) as the K dimension
    in_desc = in_desc.transform(merge(("r", "s", "c"), into="k"))

    # at use site, give it a value for each upper-level coordinate
    a_offset, a_valid = in_desc.offset(b, m=m_val, k=k_val)

This produces, at IR build time, the same SSA dataflow that a
hand-written implicit-GEMM convolution kernel would emit for the
conv-to-implicit-GEMM mapping. The kernel author never has to write
`gm // (HO * WO)` or `(ho * sH - pH + r * dH)` manually; the
descriptor algebra emits those operations and tracks the validity
predicate alongside.

Mapping to CK Tile concepts:
  - `NaiveTensorDescriptor`  ↔  `make_naive_tensor_descriptor`
  - `transform`              ↔  `transform_tensor_descriptor`
  - `pass_through`           ↔  `PassThroughTransform`
  - `pad`                    ↔  `PadTransform` (validity-only variant)
  - `merge`                  ↔  `MergeTransform`
  - `unmerge`                ↔  `UnmergeTransform`
  - `embed`                  ↔  `EmbedTransform`

The transforms compose into a DAG; calling `.offset()` walks the DAG
backwards from the upper-level coords down to the naive (lower-level)
coords, emitting one piece of SSA per transform. The same DAG can be
re-used at many call sites; each emission produces fresh SSA names.

This is the implementation surface the bake-off kernels
(`example/ck_tile/dsl/08_bake_off_implicit_gemm`,
`09_bake_off_direct_conv_16c`, `10_bake_off_direct_conv_4c`)
build their tile windows on. It is also the natural surface for
authoring new kernel families that aren't GEMM (attention, group
convolution, reductions): the same vocabulary captures every shape's
addressing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.ir import F16, I32, I64, IRBuilder, Type, Value


# ---------------------------------------------------------------------
# CoordVar: one symbolic coordinate (an SSA value + optional validity)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CoordVar:
    """One named coordinate at one level of the descriptor DAG.

    `value` is an i32 SSA `Value`. `valid` is either None (always valid)
    or an i1 SSA `Value` representing the conjunction of all
    boundary-check predicates that produced this coord.
    """

    name: str
    value: Value
    valid: Optional[Value] = None


def _and(b: IRBuilder, p: Optional[Value], q: Optional[Value]) -> Optional[Value]:
    """Conjunction of two optional i1 predicates."""
    if p is None:
        return q
    if q is None:
        return p
    return b.land(p, q)


def _ge(b: IRBuilder, lhs: Value, rhs: Value) -> Value:
    """Signed `lhs >= rhs` -> i1."""
    return b.cmp_ge(lhs, rhs)


def _lt(b: IRBuilder, lhs: Value, rhs: Value) -> Value:
    return b.cmp_lt(lhs, rhs)


# ---------------------------------------------------------------------
# Magic-number division (mirrors ck_tile/core/utility/magic_div.hpp)
# ---------------------------------------------------------------------


def calculate_magic_numbers(divisor: int) -> Tuple[int, int]:
    """Compute the ``(multiplier, shift)`` pair for unsigned magic division.

    Direct port of CK Tile's
    ``magic_division32_bit_range::calculate_magic_numbers`` (the
    ``magic_division`` alias used by ``merge_v2_magic_division``). The
    returned constants reproduce ``dividend // divisor`` via the mul-hi
    sequence in :func:`do_magic_division` for any ``dividend`` that fits
    in the 31-bit unsigned range (the same restriction CK Tile documents).

    The math:

    .. code-block:: text

        shift      = ceil(log2(divisor))           (smallest s with 2**s >= divisor)
        multiplier = ((2**shift - divisor) << 32) // divisor + 1

    Both values are plain Python ``int`` (compile-time constants); they are
    baked into the emitted IR as ``const_i32`` operands, exactly like the
    C++ constexpr path.
    """
    if divisor < 1:
        raise ValueError(f"magic division requires divisor >= 1, got {divisor}")
    shift = 0
    while (1 << shift) < divisor:
        shift += 1
    multiplier = (((1 << shift) - divisor) << 32) // divisor + 1
    return multiplier, shift


def do_magic_division(
    b: IRBuilder, dividend: Value, multiplier: int, shift: int
) -> Value:
    """Emit ``dividend // divisor`` using the magic ``(multiplier, shift)``.

    Mirrors CK Tile's ``do_magic_division`` device path
    (``tmp = __umulhi(dividend, multiplier); (tmp + dividend) >> shift``),
    which the AMDGPU backend lowers without an integer-division instruction.
    ``dividend`` is an i32 SSA value interpreted as unsigned; the result is
    the i32 quotient. ``multiplier`` / ``shift`` come from
    :func:`calculate_magic_numbers`.
    """
    # ``multiplier`` is a uint32 that may exceed the i32 signed range; bake
    # it into the i32 constant as its two's-complement bit pattern so the
    # unsigned ``umul_hi`` sees the right bits (matches the C++ uint32 path).
    mult_i32 = multiplier - (1 << 32) if multiplier >= (1 << 31) else multiplier
    tmp = b.umul_hi_i32(dividend, b.const_i32(mult_i32))
    summed = b.add(tmp, dividend)
    if shift == 0:
        return summed
    return b.lshr(summed, b.const_i32(shift))


# ---------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------


class Transform:
    """One node in the coord-transform DAG.

    Subclasses implement `apply(b, coords)` which takes the
    *upper-level* coords (the ones the kernel will provide at use
    site) and returns the *lower-level* coords (one step closer to
    the naive tensor descriptor).

    Multiple transforms compose left-to-right: the `transform()`
    method on TensorDescriptor takes a list of transforms and applies
    them in order, treating each transform's outputs as the next
    transform's inputs.
    """

    upper: Tuple[str, ...]
    """Names of the coords this transform consumes (upper level)."""

    lower: Tuple[str, ...]
    """Names of the coords this transform produces (lower level)."""

    def apply(
        self,
        b: IRBuilder,
        coords: Dict[str, CoordVar],
    ) -> Dict[str, CoordVar]:
        raise NotImplementedError

    # -- incremental (delta) lowering -------------------------------------
    #
    # ``is_linear`` marks transforms whose ``update_lower_index`` produces a
    # lower-index delta that is a function of the *upper* deltas alone (not
    # the absolute position) -- exactly CK Tile's ``IsLinearTransform`` set.
    # For those, :meth:`TensorDescriptor.move` can propagate deltas through
    # the chain and never re-touch the absolute index (so a loop-invariant
    # delta folds to a constant offset add). Non-linear transforms
    # (``Unmerge*``/``Modulo``/``XorT``) recompute their lowers from the new
    # absolute upper coords, mirroring the C++ ``update_lower_index`` for
    # those primitives.
    is_linear: bool = False

    def update_lower(
        self,
        b: IRBuilder,
        old: Dict[str, CoordVar],
        delta: Dict[str, CoordVar],
        new: Dict[str, CoordVar],
    ) -> Dict[str, CoordVar]:
        """Return the lower-coord deltas for this transform.

        Default (non-linear) path: ``new_lower - old_lower`` where each is
        obtained by :meth:`apply`. Linear transforms override this to derive
        the lower delta straight from the upper deltas, matching the C++
        ``update_lower_index`` of the corresponding primitive.
        """
        new_low = self.apply(b, new)
        old_low = self.apply(b, old)
        out: Dict[str, CoordVar] = {}
        for name in self.lower:
            d = b.sub(new_low[name].value, old_low[name].value)
            out[name] = CoordVar(name, d, new_low[name].valid)
        return out


@dataclass(frozen=True)
class PassThrough(Transform):
    """Identity rename: lower[0] = upper[0] (the coord just gets a new
    name at the lower level).

    Use when adding a transform that renames part of the input space
    while transforming another part.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]

    def __init__(self, upper_name: str, lower_name: Optional[str] = None) -> None:
        object.__setattr__(self, "upper", (upper_name,))
        object.__setattr__(self, "lower", (lower_name or upper_name,))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        return {self.lower[0]: replace(u, name=self.lower[0])}

    def update_lower(self, b, old, delta, new):
        d = delta[self.upper[0]]
        return {self.lower[0]: CoordVar(self.lower[0], d.value, None)}


@dataclass(frozen=True)
class Pad(Transform):
    """Adds `lo <= coord < hi` to the validity predicate of `coord`.

    Does *not* change the coord's value (CK Tile's PadTransform is
    "extend the descriptor with zero outside"; the value stays as
    written, the validity flips to false outside the bounds). The
    naive descriptor's load path uses the validity predicate to clamp
    the offset to a safe in-range value and zero the loaded data.

    Use to express convolution's H/W padding without touching the
    affine arithmetic (which lives in `embed`).
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    lo: int
    hi: int

    def __init__(self, coord_name: str, lo: int, hi: int) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", (coord_name,))
        object.__setattr__(self, "lo", int(lo))
        object.__setattr__(self, "hi", int(hi))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        c_lo = b.const_i32(self.lo)
        c_hi = b.const_i32(self.hi)
        valid = _and(b, _ge(b, u.value, c_lo), _lt(b, u.value, c_hi))
        merged_valid = _and(b, u.valid, valid)
        return {self.lower[0]: CoordVar(self.lower[0], u.value, merged_valid)}

    def update_lower(self, b, old, delta, new):
        # value delta passes straight through; validity is re-checked at the
        # new absolute position (validity is not a delta quantity).
        d = delta[self.upper[0]]
        valid = self.apply(b, new)[self.lower[0]].valid
        return {self.lower[0]: CoordVar(self.lower[0], d.value, valid)}


@dataclass(frozen=True)
class Embed(Transform):
    """Affine map: lower = sum_i(strides[i] * upper[i]) + offset.

    With a bound check: validity is `lo <= lower < hi` AND-ed with all
    incoming validities.

    Examples:
      - Conv `(ho, r) -> hi`:
            Embed(upper=("ho", "r"), lower="h", strides=(sH, dH),
                  offset=-pH, lo=0, hi=Hi)
        emits `h = ho * sH + r * dH - pH`, valid iff `0 <= h < Hi`.
      - Stride-only embed (no padding):
            Embed(upper=("wo", "s"), lower="w", strides=(sW, dW),
                  offset=0, lo=0, hi=Wi)
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    strides: Tuple[int, ...]
    offset: int
    lo: int
    hi: int

    def __init__(
        self,
        upper: Sequence[str],
        lower: str,
        strides: Sequence[int],
        offset: int = 0,
        lo: Optional[int] = None,
        hi: Optional[int] = None,
    ) -> None:
        if len(upper) != len(strides):
            raise ValueError(
                f"Embed expects len(upper) == len(strides) (got {upper!r}, {strides!r})"
            )
        object.__setattr__(self, "upper", tuple(upper))
        object.__setattr__(self, "lower", (lower,))
        object.__setattr__(self, "strides", tuple(int(s) for s in strides))
        object.__setattr__(self, "offset", int(offset))
        object.__setattr__(self, "lo", -(1 << 30) if lo is None else int(lo))
        object.__setattr__(self, "hi", (1 << 30) if hi is None else int(hi))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        # acc = offset + sum(strides[i] * coords[upper[i]])
        acc: Optional[Value] = None
        valid_acc: Optional[Value] = None
        for name, s in zip(self.upper, self.strides):
            u = coords[name]
            valid_acc = _and(b, valid_acc, u.valid)
            if s == 1:
                term = u.value
            else:
                term = b.mul(u.value, b.const_i32(s))
            acc = term if acc is None else b.add(acc, term)
        if self.offset != 0:
            acc = b.add(acc, b.const_i32(self.offset))
        if acc is None:
            acc = b.const_i32(self.offset)
        # bounds: lo <= acc < hi
        bounds = _and(
            b,
            _ge(b, acc, b.const_i32(self.lo)),
            _lt(b, acc, b.const_i32(self.hi)),
        )
        valid = _and(b, valid_acc, bounds)
        return {self.lower[0]: CoordVar(self.lower[0], acc, valid)}

    is_linear = True

    def update_lower(self, b, old, delta, new):
        # delta_low = sum(strides[i] * delta_up[i]); constant offset drops out.
        acc: Optional[Value] = None
        for name, s in zip(self.upper, self.strides):
            d = delta[name].value
            term = d if s == 1 else b.mul(d, b.const_i32(s))
            acc = term if acc is None else b.add(acc, term)
        if acc is None:
            acc = b.const_i32(0)
        valid = self.apply(b, new)[self.lower[0]].valid
        return {self.lower[0]: CoordVar(self.lower[0], acc, valid)}


@dataclass(frozen=True)
class Merge(Transform):
    """Flatten N upper coords into one linear lower coord.

    lower = upper[0]*D1*D2*...*D_{N-1} + upper[1]*D2*...*D_{N-1} + ... + upper[N-1]

    where `dims = (D0, D1, ..., D_{N-1})` are the *upper bound* sizes
    of each upper coord. Note that `dims[0]` is not used in the math
    (it's the bound of the leading coord) but the merge expects it
    for symmetry with `Unmerge`.

    Use to package the implicit-GEMM `m = n*Ho*Wo + ho*Wo + wo` and
    `k = r*S*C + s*C + c` mappings.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    dims: Tuple[int, ...]

    def __init__(self, upper: Sequence[str], into: str, dims: Sequence[int]) -> None:
        if len(upper) != len(dims):
            raise ValueError(
                f"Merge expects len(upper) == len(dims) (got {upper!r}, {dims!r})"
            )
        object.__setattr__(self, "upper", tuple(upper))
        object.__setattr__(self, "lower", (into,))
        object.__setattr__(self, "dims", tuple(int(d) for d in dims))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        # Stride for coord i = product of dims[i+1:].
        names = self.upper
        dims = self.dims
        acc: Optional[Value] = None
        valid: Optional[Value] = None
        for i, name in enumerate(names):
            u = coords[name]
            valid = _and(b, valid, u.valid)
            stride = 1
            for d in dims[i + 1 :]:
                stride *= d
            if stride == 1:
                term = u.value
            else:
                term = b.mul(u.value, b.const_i32(stride))
            acc = term if acc is None else b.add(acc, term)
        if acc is None:
            acc = b.const_i32(0)
        return {self.lower[0]: CoordVar(self.lower[0], acc, valid)}

    is_linear = True

    def update_lower(self, b, old, delta, new):
        # delta_low = sum(stride_i * delta_up_i); same linear map as apply.
        acc: Optional[Value] = None
        for i, name in enumerate(self.upper):
            stride = 1
            for d in self.dims[i + 1 :]:
                stride *= d
            dv = delta[name].value
            term = dv if stride == 1 else b.mul(dv, b.const_i32(stride))
            acc = term if acc is None else b.add(acc, term)
        if acc is None:
            acc = b.const_i32(0)
        return {self.lower[0]: CoordVar(self.lower[0], acc, None)}


@dataclass(frozen=True)
class Unmerge(Transform):
    """Inverse of `Merge`: split a flat coord into N independent coords.

    upper / D1*D2*...*D_{N-1}                          -> lower[0]
    (upper / D2*...*D_{N-1}) % D1                      -> lower[1]
    ...
    upper % D_{N-1}                                    -> lower[N-1]

    Use to recover (n, ho, wo) from the implicit-GEMM m, or (r, s, c)
    from k, the way a hand-written implicit-GEMM kernel does manually.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    dims: Tuple[int, ...]

    def __init__(
        self, upper_name: str, lowers: Sequence[str], dims: Sequence[int]
    ) -> None:
        if len(lowers) != len(dims):
            raise ValueError(
                f"Unmerge expects len(lowers) == len(dims) (got {lowers!r}, {dims!r})"
            )
        object.__setattr__(self, "upper", (upper_name,))
        object.__setattr__(self, "lower", tuple(lowers))
        object.__setattr__(self, "dims", tuple(int(d) for d in dims))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        out: Dict[str, CoordVar] = {}
        for i, name in enumerate(self.lower):
            stride = 1
            for d in self.dims[i + 1 :]:
                stride *= d
            if stride == 1:
                quot = u.value
            else:
                quot = b.div(u.value, b.const_i32(stride))
            if i == 0:
                # leading coord: just the quotient (no modulo needed)
                val = quot
            else:
                # interior + last: modulo dims[i]
                val = b.mod(quot, b.const_i32(self.dims[i]))
            out[name] = CoordVar(name, val, u.valid)
        return out


@dataclass(frozen=True)
class UnmergeMagicDiv(Transform):
    """``Unmerge`` that splits via magic-number division (no hardware div).

    Numerically identical to :class:`Unmerge` -- it splits one flat upper
    coord into ``N`` lower coords -- but emits the mul-hi magic-division
    sequence (:func:`do_magic_division`) instead of literal ``b.div`` /
    ``b.mod``. This mirrors CK Tile's ``merge_v2_magic_division`` (the
    default ``make_merge_transform``), whose ``calculate_lower_index`` is
    exactly this flat-to-multi split. It removes the integer-division /
    modulo ops from the runtime addressing path (the single biggest source
    of div/mod assembly in conv ``m -> (n, ho, wo)`` / ``k -> (r, s, c)``
    decodes).

    Following the C++ algorithm, the split walks from the *last* lower coord
    to the first: ``tmp = idx_up``; for ``i`` from ``N-1`` down to ``1``,
    ``q = tmp // dims[i]`` (magic), ``lower[i] = tmp - q*dims[i]``,
    ``tmp = q``; finally ``lower[0] = tmp``. The dividend must stay within
    the documented 31-bit unsigned range.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    dims: Tuple[int, ...]

    def __init__(
        self, upper_name: str, lowers: Sequence[str], dims: Sequence[int]
    ) -> None:
        if len(lowers) != len(dims):
            raise ValueError(
                f"UnmergeMagicDiv expects len(lowers) == len(dims) "
                f"(got {lowers!r}, {dims!r})"
            )
        object.__setattr__(self, "upper", (upper_name,))
        object.__setattr__(self, "lower", tuple(lowers))
        object.__setattr__(self, "dims", tuple(int(d) for d in dims))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        n = len(self.lower)
        out: Dict[str, CoordVar] = {}
        tmp = u.value
        # Walk last -> 1, peeling off the remainder against each dim.
        for i in range(n - 1, 0, -1):
            d = self.dims[i]
            if d == 1:
                # x // 1 == x, x % 1 == 0; no magic needed.
                rem = b.const_i32(0)
                quot = tmp
            else:
                mult, shift = calculate_magic_numbers(d)
                quot = do_magic_division(b, tmp, mult, shift)
                rem = b.sub(tmp, b.mul(quot, b.const_i32(d)))
            name = self.lower[i]
            out[name] = CoordVar(name, rem, u.valid)
            tmp = quot
        out[self.lower[0]] = CoordVar(self.lower[0], tmp, u.valid)
        return out


@dataclass(frozen=True)
class UnmergeDivMod(Transform):
    """``Unmerge`` variant emitting the pow-2 division/mod scan.

    Mirrors CK Tile's ``merge_v3_division_mod`` (the literal ``/`` and
    ``%`` variant intended for compile-time, power-of-two ``low_lengths``).
    It uses the *reverse-exclusive-scan* of the dims (the product of all
    dims to the right of ``i``) as the divisor, walking the lowers from
    first to last:

    .. code-block:: text

        tmp = idx_up
        for i in 0 .. N-2:   lower[i] = tmp // scan[i]; tmp %= scan[i]
        lower[N-1] = tmp

    Numerically identical to :class:`Unmerge`, just expressed in the
    scan form the C++ ``merge_v3`` uses; provided for parity with the
    cshuffle-epilogue LDS descriptor chain, which names this transform.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    dims: Tuple[int, ...]

    def __init__(
        self, upper_name: str, lowers: Sequence[str], dims: Sequence[int]
    ) -> None:
        if len(lowers) != len(dims):
            raise ValueError(
                f"UnmergeDivMod expects len(lowers) == len(dims) "
                f"(got {lowers!r}, {dims!r})"
            )
        object.__setattr__(self, "upper", (upper_name,))
        object.__setattr__(self, "lower", tuple(lowers))
        object.__setattr__(self, "dims", tuple(int(d) for d in dims))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        n = len(self.lower)
        out: Dict[str, CoordVar] = {}
        tmp = u.value
        for i in range(n - 1):
            scan = 1
            for d in self.dims[i + 1 :]:
                scan *= d
            name = self.lower[i]
            if scan == 1:
                out[name] = CoordVar(name, tmp, u.valid)
                # tmp %= 1 == 0
                tmp = b.const_i32(0)
            else:
                c_scan = b.const_i32(scan)
                out[name] = CoordVar(name, b.div(tmp, c_scan), u.valid)
                tmp = b.mod(tmp, c_scan)
        out[self.lower[-1]] = CoordVar(self.lower[-1], tmp, u.valid)
        return out


@dataclass(frozen=True)
class XorT(Transform):
    """2-D XOR swizzle: ``lower[1] = upper[1] ^ (upper[0] % length1)``.

    Composable port of CK Tile's ``xor_t`` transform
    (``coordinate_transform.hpp``). The first coord passes through
    (``lower[0] = upper[0]``); the second is XOR-ed with the first taken
    modulo ``length1`` (the lower length of the second dim). With
    ``apply_modulo=False`` the raw ``upper[0]`` is used (the C++
    ``ApplyModulo=false`` branch). This is the LDS bank-conflict-avoidance
    swizzle expressed inside the descriptor chain, the general counterpart
    of the closed-form byte table in ``helpers/layouts.py``.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    length1: int
    apply_modulo: bool

    def __init__(
        self,
        upper: Sequence[str],
        lowers: Sequence[str],
        *,
        length1: int,
        apply_modulo: bool = True,
    ) -> None:
        if len(upper) != 2 or len(lowers) != 2:
            raise ValueError(f"XorT is 2->2 (got upper={upper!r}, lowers={lowers!r})")
        object.__setattr__(self, "upper", tuple(upper))
        object.__setattr__(self, "lower", tuple(lowers))
        object.__setattr__(self, "length1", int(length1))
        object.__setattr__(self, "apply_modulo", bool(apply_modulo))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u0 = coords[self.upper[0]]
        u1 = coords[self.upper[1]]
        valid = _and(b, u0.valid, u1.valid)
        if self.apply_modulo and self.length1 != 1:
            swz = b.mod(u0.value, b.const_i32(self.length1))
        elif self.apply_modulo:
            # x % 1 == 0 -> xor by 0 is identity
            swz = b.const_i32(0)
        else:
            swz = u0.value
        low1 = b.xor(u1.value, swz)
        return {
            self.lower[0]: CoordVar(self.lower[0], u0.value, u0.valid),
            self.lower[1]: CoordVar(self.lower[1], low1, valid),
        }


@dataclass(frozen=True)
class Slice(Transform):
    """Window into a dim: ``lower = upper + begin`` (``begin <= lower < end``).

    Port of CK Tile's ``slice`` transform: the upper coord ranges over
    ``[0, end - begin)`` and maps to ``[begin, end)`` in the lower space by
    adding the compile-time ``begin``. Use for conv group/batch slicing
    that today is done with ad-hoc offset adds. (CK Tile's ``slice`` carries
    no boundary check on its own; the enclosing tile window bounds it.)
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    begin: int
    end: int

    def __init__(
        self, coord_name: str, *, begin: int, end: int, into: Optional[str] = None
    ) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", (into or coord_name,))
        object.__setattr__(self, "begin", int(begin))
        object.__setattr__(self, "end", int(end))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        val = u.value if self.begin == 0 else b.add(u.value, b.const_i32(self.begin))
        return {self.lower[0]: CoordVar(self.lower[0], val, u.valid)}

    def update_lower(self, b, old, delta, new):
        d = delta[self.upper[0]]
        return {self.lower[0]: CoordVar(self.lower[0], d.value, None)}


@dataclass(frozen=True)
class Freeze(Transform):
    """Pin a lower dim to a constant, consuming no upper coord.

    Port of CK Tile's ``freeze`` (1 lower, 0 upper): the lower coord is set
    to a fixed compile-time index ``low_idx`` regardless of any upper coord.
    Use to nail a batch/group dimension to a constant inside a descriptor
    chain (the "pin a dim to a constant" idiom done with offset adds today).
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    low_idx: int

    def __init__(self, into: str, *, low_idx: int) -> None:
        object.__setattr__(self, "upper", ())
        object.__setattr__(self, "lower", (into,))
        object.__setattr__(self, "low_idx", int(low_idx))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        return {self.lower[0]: CoordVar(self.lower[0], b.const_i32(self.low_idx), None)}

    def update_lower(self, b, old, delta, new):
        # frozen: lower never moves -> zero delta.
        return {self.lower[0]: CoordVar(self.lower[0], b.const_i32(0), None)}


@dataclass(frozen=True)
class Insert(Transform):
    """Add a dangling upper dim with no lower dim (consumes the coord).

    Port of CK Tile's ``insert`` (0 lower, 1 upper): the upper coord exists
    in the upper space (it has a length) but contributes nothing to the
    lower addressing. Used to introduce a broadcast / iteration dim that the
    naive layout does not see.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    length: int

    def __init__(self, coord_name: str, *, length: int) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", ())
        object.__setattr__(self, "length", int(length))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        # 0 lower dims: the upper coord is consumed but produces nothing.
        return {}


@dataclass(frozen=True)
class Replicate(Transform):
    """Add N dangling upper dims with no lower dim (broadcast).

    Port of CK Tile's ``replicate`` (0 lower, N upper): every upper coord
    exists for iteration but contributes nothing to lower addressing, so
    the same lower element is read for each value -- the descriptor-level
    broadcast. Used for paged-KV / norm broadcast dims.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    lengths: Tuple[int, ...]

    def __init__(self, uppers: Sequence[str], *, lengths: Sequence[int]) -> None:
        if len(uppers) != len(lengths):
            raise ValueError(
                f"Replicate expects len(uppers) == len(lengths) "
                f"(got {uppers!r}, {lengths!r})"
            )
        object.__setattr__(self, "upper", tuple(uppers))
        object.__setattr__(self, "lower", ())
        object.__setattr__(self, "lengths", tuple(int(x) for x in lengths))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        return {}


@dataclass(frozen=True)
class Modulo(Transform):
    """Wrap a dim: ``lower = upper % modulus``.

    Port of CK Tile's ``modulo`` transform. Use for ring-buffer / circular
    addressing (paged-KV slot wrap). The upper coord ranges over
    ``[0, up_length)`` and the lower is its residue mod ``modulus``.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    modulus: int

    def __init__(
        self, coord_name: str, *, modulus: int, into: Optional[str] = None
    ) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", (into or coord_name,))
        object.__setattr__(self, "modulus", int(modulus))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        if self.modulus == 1:
            val = b.const_i32(0)
        else:
            val = b.mod(u.value, b.const_i32(self.modulus))
        return {self.lower[0]: CoordVar(self.lower[0], val, u.valid)}


@dataclass(frozen=True)
class Offset(Transform):
    """Shift a dim by a constant: ``lower = upper + offset_length``.

    Port of CK Tile's ``offset`` transform (the index-shifting cousin of
    the rocke validity-only :class:`Pad`). Unlike :class:`Slice` (whose
    upper length shrinks to ``end - begin``), ``offset`` keeps the same
    length and merely biases the index, as CK Tile's ``offset`` does.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    offset_length: int

    def __init__(
        self, coord_name: str, *, offset_length: int, into: Optional[str] = None
    ) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", (into or coord_name,))
        object.__setattr__(self, "offset_length", int(offset_length))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        val = (
            u.value
            if self.offset_length == 0
            else b.add(u.value, b.const_i32(self.offset_length))
        )
        return {self.lower[0]: CoordVar(self.lower[0], val, u.valid)}

    def update_lower(self, b, old, delta, new):
        d = delta[self.upper[0]]
        return {self.lower[0]: CoordVar(self.lower[0], d.value, None)}


@dataclass(frozen=True)
class RightPad(Transform):
    """Index-shifting right pad: value passes through, valid iff ``< low_length``.

    Port of CK Tile's ``right_pad`` transform. The lower index equals the
    upper index (``calculate_lower_index`` is identity); the validity check
    is ``idx_up < low_length`` (the unpadded extent). The upper extent is
    ``low_length + right_pad_length`` and indices in the pad region are the
    out-of-range ones. This is the index-shifting pad CK Tile uses (the
    rocke :class:`Pad` is validity-only with explicit ``lo``/``hi``).
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    low_length: int

    def __init__(self, coord_name: str, *, low_length: int) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", (coord_name,))
        object.__setattr__(self, "low_length", int(low_length))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        valid = _and(b, u.valid, _lt(b, u.value, b.const_i32(self.low_length)))
        return {self.lower[0]: CoordVar(self.lower[0], u.value, valid)}

    def update_lower(self, b, old, delta, new):
        d = delta[self.upper[0]]
        valid = self.apply(b, new)[self.lower[0]].valid
        return {self.lower[0]: CoordVar(self.lower[0], d.value, valid)}


@dataclass(frozen=True)
class LeftPad(Transform):
    """Index-shifting left pad: ``lower = upper - left_pad``, valid iff ``>= left_pad``.

    Port of CK Tile's ``left_pad`` transform: the lower index is shifted
    down by ``left_pad_length`` (``calculate_lower_index`` subtracts it),
    and the validity check is ``idx_up >= left_pad_length``. The upper
    extent is ``low_length + left_pad_length``. This is the value-shifting
    pad variant CK Tile uses, distinct from the validity-only :class:`Pad`.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    left_pad: int

    def __init__(
        self, coord_name: str, *, left_pad: int, into: Optional[str] = None
    ) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", (into or coord_name,))
        object.__setattr__(self, "left_pad", int(left_pad))

    is_linear = True

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        c_lp = b.const_i32(self.left_pad)
        val = u.value if self.left_pad == 0 else b.sub(u.value, c_lp)
        valid = _and(b, u.valid, _ge(b, u.value, c_lp))
        return {self.lower[0]: CoordVar(self.lower[0], val, valid)}

    def update_lower(self, b, old, delta, new):
        # lower shifts by a constant -> value delta == upper delta.
        d = delta[self.upper[0]]
        valid = self.apply(b, new)[self.lower[0]].valid
        return {self.lower[0]: CoordVar(self.lower[0], d.value, valid)}


# Convenience constructors so user code reads like the C++ DSL.


def pass_through(coord: str, into: Optional[str] = None) -> PassThrough:
    return PassThrough(coord, into)


def pad(coord: str, *, lo: int, hi: int) -> Pad:
    return Pad(coord, lo=lo, hi=hi)


def embed(
    upper: Sequence[str],
    into: str,
    *,
    strides: Sequence[int],
    offset: int = 0,
    lo: Optional[int] = None,
    hi: Optional[int] = None,
) -> Embed:
    return Embed(upper, into, strides=strides, offset=offset, lo=lo, hi=hi)


def merge(upper: Sequence[str], *, into: str, dims: Sequence[int]) -> Merge:
    return Merge(upper, into, dims)


def unmerge(upper: str, into: Sequence[str], *, dims: Sequence[int]) -> Unmerge:
    return Unmerge(upper, into, dims)


def merge_magic(upper: Sequence[str], *, into: str, dims: Sequence[int]) -> Merge:
    """``merge`` (flatten N coords -> 1).

    CK Tile's ``merge`` (the flatten direction) is a pure linear
    combination -- ``merge_v2_magic_division`` only differs from the literal
    ``merge`` in the *lowering* (split) direction, which in rocke is
    :func:`unmerge_magic`. So ``merge_magic`` is the same arithmetic as
    :func:`merge` and is provided as the symmetric name; the magic-division
    win lives entirely in :func:`unmerge_magic`.
    """
    return Merge(upper, into, dims)


def unmerge_magic(
    upper: str, into: Sequence[str], *, dims: Sequence[int]
) -> UnmergeMagicDiv:
    """``unmerge`` (split 1 -> N) via magic-number division.

    Drop-in replacement for :func:`unmerge` that emits the mul-hi
    magic-division sequence instead of literal ``div``/``mod`` (CK Tile's
    ``merge_v2_magic_division``). Use in any K-looping / sliding addressing
    decode where the divisors are loop-invariant compile-time constants.
    """
    return UnmergeMagicDiv(upper, into, dims)


def unmerge_div_mod(
    upper: str, into: Sequence[str], *, dims: Sequence[int]
) -> UnmergeDivMod:
    """``unmerge`` via the pow-2 division/mod scan (CK Tile ``merge_v3``)."""
    return UnmergeDivMod(upper, into, dims)


def xor_t(
    upper: Sequence[str],
    into: Sequence[str],
    *,
    length1: int,
    apply_modulo: bool = True,
) -> XorT:
    """2-D XOR swizzle ``lower[1] = upper[1] ^ (upper[0] % length1)``."""
    return XorT(upper, into, length1=length1, apply_modulo=apply_modulo)


def slice_(coord: str, *, begin: int, end: int, into: Optional[str] = None) -> Slice:
    """Window a dim to ``[begin, end)`` (``lower = upper + begin``)."""
    return Slice(coord, begin=begin, end=end, into=into)


def freeze(into: str, *, low_idx: int) -> Freeze:
    """Pin a lower dim to the constant ``low_idx`` (consumes no upper coord)."""
    return Freeze(into, low_idx=low_idx)


def insert(coord: str, *, length: int) -> Insert:
    """Add a dangling upper dim with no lower dim."""
    return Insert(coord, length=length)


def replicate(uppers: Sequence[str], *, lengths: Sequence[int]) -> Replicate:
    """Add N dangling upper dims (broadcast: no lower contribution)."""
    return Replicate(uppers, lengths=lengths)


def modulo(coord: str, *, modulus: int, into: Optional[str] = None) -> Modulo:
    """Wrap a dim: ``lower = upper % modulus``."""
    return Modulo(coord, modulus=modulus, into=into)


def offset(coord: str, *, offset_length: int, into: Optional[str] = None) -> Offset:
    """Shift a dim by a constant ``lower = upper + offset_length``."""
    return Offset(coord, offset_length=offset_length, into=into)


def right_pad(coord: str, *, low_length: int) -> RightPad:
    """Index-shifting right pad: valid iff ``upper < low_length``."""
    return RightPad(coord, low_length=low_length)


def left_pad(coord: str, *, left_pad: int, into: Optional[str] = None) -> LeftPad:
    """Index-shifting left pad: ``lower = upper - left_pad``, valid iff ``>=``."""
    return LeftPad(coord, left_pad=left_pad, into=into)


# ---------------------------------------------------------------------
# Runtime-bound transforms
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PadDynamic(Transform):
    """``pad`` with one or both bounds supplied as runtime SSA values.

    CK Tile's ``PadTransform`` accepts both compile-time and runtime
    bounds; the original :class:`Pad` here only handles the compile-
    time case. ``PadDynamic`` lifts that restriction: ``lo`` and
    ``hi`` may be either ``None`` (skip that side), an ``int``
    (compile-time), or an ``i32 SSA Value`` (runtime, e.g. the
    ``cur_batch_q_len`` or ``NUM_QH`` that attention kernels test
    against). The coord's value passes through unchanged; the
    validity predicate picks up ``lo <= value < hi`` AND-ed with any
    incoming validity.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]
    lo: Any
    hi: Any

    def __init__(
        self,
        coord_name: str,
        *,
        lo: Any = None,
        hi: Any = None,
    ) -> None:
        object.__setattr__(self, "upper", (coord_name,))
        object.__setattr__(self, "lower", (coord_name,))
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @staticmethod
    def _as_value(b: IRBuilder, x: Any) -> Value:
        if isinstance(x, Value):
            return x
        return b.const_i32(int(x))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        bounds: Optional[Value] = None
        if self.lo is not None:
            bounds = _and(b, bounds, _ge(b, u.value, self._as_value(b, self.lo)))
        if self.hi is not None:
            bounds = _and(b, bounds, _lt(b, u.value, self._as_value(b, self.hi)))
        merged_valid = _and(b, u.valid, bounds)
        return {self.lower[0]: CoordVar(self.lower[0], u.value, merged_valid)}


def pad_dynamic(coord: str, *, lo: Any = None, hi: Any = None) -> PadDynamic:
    """``pad`` with possibly-runtime bounds.

    Pass ``lo`` / ``hi`` as ``int`` for compile-time bounds, as an
    ``i32 SSA Value`` for runtime bounds, or ``None`` to skip that
    side. Use this when the upper bound only becomes known at launch
    time (e.g. ``cur_batch_q_len`` in attention) or when only one
    side of the range needs a check.
    """
    return PadDynamic(coord, lo=lo, hi=hi)


@dataclass(frozen=True)
class Indirect(Transform):
    """Table-lookup transform: ``physical = table[base + upper]``.

    Implements CK Tile's "indirection" pattern for paged-KV style
    addressing where a *logical* index is mapped to a *physical*
    block via a one-dimensional lookup table. Concretely:

    .. code-block:: text

        idx        = base + upper
        physical   = global_load_i32(table, idx)

    The upper coord (e.g. ``tile_idx`` for paged attention) is
    consumed; the lower coord (``physical_block``) is what subsequent
    transforms see. ``base`` is the offset within the table (often
    ``seq_idx * block_table_stride`` for multi-sequence batches);
    ``table`` is the i32* pointer; the load is unguarded so callers
    are responsible for keeping the indexed offset in range.

    This is the descriptor-level equivalent of
    :meth:`PagedKvDescriptor.block_base_from_table` and lets the
    whole paged-KV addressing collapse into one descriptor chain.
    """

    upper: Tuple[str, ...]
    lower: Tuple[str, ...]

    def __init__(
        self,
        upper_name: str,
        *,
        lower_name: str,
        table: Value,
        base: Value,
        max_idx: Optional[Value] = None,
        default_value: int = 0,
    ) -> None:
        object.__setattr__(self, "upper", (upper_name,))
        object.__setattr__(self, "lower", (lower_name,))
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_max_idx", max_idx)
        object.__setattr__(self, "_default_value", int(default_value))

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        idx = b.add(self._base, u.value)
        if self._max_idx is None:
            # Unguarded: caller promises ``base + upper`` is in range. Kept
            # for the call sites that have stronger invariants than the
            # rounded-up multi-block tile (they read exactly
            # ``ceil(kv_len / bs)`` entries per seq).
            physical = b.global_load_i32(self._table, idx)
        else:
            # OOB-safe load: when the per-seq block-table footprint is
            # smaller than what the multi-block tile chain over-fetches
            # (paged-attention with ``T > kv_len[i]`` or ``T > bs *
            # ceil(kv_len[i] / bs)``), the chain can synthesise indices
            # past the tensor end. Clamp them with a regular masked
            # load returning ``default_value`` so the downstream
            # ``global_load`` lands on a guaranteed-valid block.
            mask = b.cmp_lt(idx, self._max_idx)
            physical = b.masked_global_load(
                self._table,
                idx,
                mask,
                b.const_i32(self._default_value),
                dtype=I32,
                align=4,
            )
        return {self.lower[0]: CoordVar(self.lower[0], physical, u.valid)}


def indirect(
    upper: str,
    *,
    into: str,
    table: Value,
    base: Value,
    max_idx: Optional[Value] = None,
    default_value: int = 0,
) -> Indirect:
    """One-line page-table lookup transform.

    Equivalent to writing the chain
    ``embed(("seq", "tile"), into="idx", strides=(stride, 1))``
    followed by an ad-hoc ``global_load_i32`` -- but explicit about
    the indirection step so legality / explainer tooling can see it.

    When ``max_idx`` is supplied the lookup uses ``masked_global_load``
    that clamps OOB indices to ``default_value`` (default: 0). This is
    needed for paged attention when the kernel's KV-tile loop
    over-fetches block-table entries past the per-seq footprint (e.g.
    short ``kv_len`` with ``T > kv_len``); the over-fetched index
    otherwise reads adjacent (uninitialised) memory and the downstream
    ``buffer_load`` can fault on an unmapped page.
    """
    return Indirect(
        upper,
        lower_name=into,
        table=table,
        base=base,
        max_idx=max_idx,
        default_value=default_value,
    )


# ---------------------------------------------------------------------
# TensorDescriptor: a chain of transforms over a naive layout
# ---------------------------------------------------------------------


@dataclass
class TensorDescriptor:
    """A chain of coordinate transforms terminating in a naive layout.

    Constructed via:
        d = TensorDescriptor.naive("A", lengths=[N, H, W, C], dtype=F16)

    The "naive" layout assumes row-major: `offset = c + W*(w + Wo*(h + Ho*n))`.
    Strides default to the row-major product of lengths to the right.
    For non-row-major layouts pass `strides=...` explicitly.

    Then apply transforms left-to-right:
        d = d.transform(unmerge("m", into=("n","ho","wo"), dims=(N, Ho, Wo)))

    Each `.transform(t1, t2, ...)` creates a *new* descriptor whose
    upper-level coordinate space is the union of all new lowers
    (those that don't appear as inputs to any of the supplied
    transforms). The old upper-level coords disappear from the new
    space.

    Finally use `desc.offset(b, m=m_val, k=k_val)` to compute an
    `(i32_offset, optional_i1_valid)` pair.
    """

    name: str
    # The "naive" base coord names + their bounds (for default valid)
    base_names: Tuple[str, ...]
    base_lengths: Tuple[int, ...]
    base_strides: Tuple[int, ...]
    # The chain of transforms, in order from naive (closest to base)
    # to upper (closest to user). When computing offset(), we start
    # with user coords and walk through transforms in reverse order.
    chain: Tuple[Transform, ...] = ()
    # The user-facing coord names at the current top of the chain
    upper_names: Tuple[str, ...] = ()

    @classmethod
    def naive(
        cls,
        name: str,
        *,
        lengths: Sequence[int],
        dtype: Type = F16,
        strides: Optional[Sequence[int]] = None,
        coord_names: Optional[Sequence[str]] = None,
    ) -> "TensorDescriptor":
        if not lengths:
            raise ValueError("naive descriptor needs at least one dim")
        lengths = tuple(int(x) for x in lengths)
        if strides is None:
            # row-major
            ss: List[int] = [1]
            for d in reversed(lengths[1:]):
                ss.insert(0, ss[0] * d)
            strides = tuple(ss[: len(lengths)])
        else:
            strides = tuple(int(x) for x in strides)
        if coord_names is None:
            coord_names = tuple(f"d{i}" for i in range(len(lengths)))
        coord_names = tuple(coord_names)
        if len(coord_names) != len(lengths):
            raise ValueError("coord_names length mismatch")
        return cls(
            name=name,
            base_names=coord_names,
            base_lengths=lengths,
            base_strides=strides,
            chain=(),
            upper_names=coord_names,
        )

    @property
    def dtype(self) -> Type:
        return F16  # for now; could be parametric

    def transform(self, *transforms: Transform) -> "TensorDescriptor":
        r"""Apply transforms in order, producing a new descriptor.

        After applying a chain of transforms, the user-facing coord
        space (`upper_names`) is computed from scratch by:

            upper_names = (base_names ∪ all_uppers) \ all_lowers

        where `all_uppers` and `all_lowers` are the union of every
        transform's `.upper` and `.lower` respectively. This is
        coord-flow-from-user-to-naive semantics: a coord that appears
        as a `lower` of some transform was consumed inside the chain,
        so the user does not supply it. A coord that appears as an
        `upper` of some transform but never as a `lower` of any
        transform is user-facing. The naive descriptor's `base_names`
        are also user-facing UNLESS they appear as the `lower` of
        some transform (in which case the chain produces them
        internally).
        """
        if not transforms:
            return self
        new_chain = self.chain + tuple(transforms)
        all_uppers = set()
        all_lowers = set()
        for t in new_chain:
            all_uppers.update(t.upper)
            all_lowers.update(t.lower)
        upper_set = (set(self.base_names) | all_uppers) - all_lowers
        # Preserve a stable order: walk base_names first, then any
        # remaining names in the order they appeared as transform
        # uppers.
        ordered: List[str] = []
        seen: set[str] = set()
        for n in self.base_names:
            if n in upper_set and n not in seen:
                ordered.append(n)
                seen.add(n)
        for t in new_chain:
            for n in t.upper:
                if n in upper_set and n not in seen:
                    ordered.append(n)
                    seen.add(n)
        return replace(self, chain=new_chain, upper_names=tuple(ordered))

    def unmerge_lower(
        self,
        b: IRBuilder,
        **upper_values: Value,
    ) -> Dict[str, Value]:
        """Run the descriptor's transform chain in topological order
        and return the lowered coordinate map.

        Equivalent to :meth:`offset`'s prologue but stops before the
        final ``base_strides`` reduction: callers get the dict of
        ``{lower_name: i32 SSA value}`` for every coord produced by
        any applicable transform plus the original upper coords.

        This is the public accessor for the ``Unmerge``-driven
        ``(n, ho, wo) <- m`` decomposition that conv kernels do
        inline today (P56). Calling
        ``desc.unmerge_lower(b, m=...)`` lets conv ``chunk_meta``
        replace its inline ``b.div`` / ``b.mod`` chain with one
        descriptor call.
        """
        coords: Dict[str, CoordVar] = {
            name: CoordVar(name, val) for name, val in upper_values.items()
        }
        remaining: List[Transform] = list(self.chain)
        while remaining:
            progress = False
            next_remaining: List[Transform] = []
            for t in remaining:
                if all(name in coords for name in t.upper):
                    produced = t.apply(b, coords)
                    for n, v in produced.items():
                        coords[n] = v
                    progress = True
                else:
                    next_remaining.append(t)
            remaining = next_remaining
            if not progress:
                # No more applicable transforms — return whatever we
                # produced so far. Callers asking for partial lowering
                # of a sub-chain (e.g. only one ``Unmerge``) get a
                # complete answer for the coords that were producible.
                break
        return {name: cv.value for name, cv in coords.items()}

    def _run_chain(
        self, b: IRBuilder, upper_values: Dict[str, Value]
    ) -> Dict[str, "CoordVar"]:
        """Resolve the transform chain to the base coords (topological)."""
        missing = set(self.upper_names) - set(upper_values.keys())
        if missing:
            raise ValueError(
                f"offset() missing upper coords for descriptor {self.name!r}: "
                f"{sorted(missing)}"
            )
        coords: Dict[str, CoordVar] = {
            name: CoordVar(name, val) for name, val in upper_values.items()
        }
        remaining: List[Transform] = list(self.chain)
        while remaining:
            progress = False
            next_remaining: List[Transform] = []
            for t in remaining:
                if all(name in coords for name in t.upper):
                    produced = t.apply(b, coords)
                    for n, v in produced.items():
                        coords[n] = v
                    progress = True
                else:
                    next_remaining.append(t)
            if not progress:
                names = [t.upper for t in next_remaining]
                avail = sorted(coords.keys())
                raise ValueError(
                    f"transform chain has unresolved deps: pending uppers "
                    f"= {names}; available coords = {avail}"
                )
            remaining = next_remaining
        return coords

    def offset(
        self,
        b: IRBuilder,
        **upper_values: Value,
    ) -> Tuple[Value, Optional[Value]]:
        """Compute (offset, valid) for the supplied upper-level coord values.

        `upper_values` maps `upper_names` -> i32 SSA values. Returns
        the linear i32 offset into the naive tensor (in elements, not
        bytes) and the i1 in-bounds predicate (or None if no
        boundary check is in flight).
        """
        coords = self._run_chain(b, upper_values)
        offset: Optional[Value] = None
        valid: Optional[Value] = None
        for name, stride in zip(self.base_names, self.base_strides):
            if name not in coords:
                raise ValueError(
                    f"after chain, base coord {name!r} not in {sorted(coords.keys())}"
                )
            c = coords[name]
            valid = _and(b, valid, c.valid)
            if stride == 1:
                term = c.value
            else:
                term = b.mul(c.value, b.const_i32(stride))
            offset = term if offset is None else b.add(offset, term)
        if offset is None:
            offset = b.const_i32(0)
        return offset, valid

    def offset_i64_split(
        self,
        b: IRBuilder,
        base_coord: str,
        **upper_values: Value,
    ) -> Tuple[Value, Value, Optional[Value]]:
        """Like :meth:`offset` but returns ``(base_i64, within_i32, valid)``.

        ``base_i64`` is the ``base_coord`` term computed in **i64** (so it
        does not overflow the i32 offset for paged caches > 2 GiB);
        ``within_i32`` is the sum of all the other (small) base terms. A
        paged-KV load can put ``base_i64`` into a 64-bit buffer base and
        keep ``within_i32`` in the 32-bit buffer voffset. ``base_coord``'s
        value must be wave-uniform for the buffer base to stay scalar.
        """
        coords = self._run_chain(b, upper_values)
        base_i64: Optional[Value] = None
        within: Optional[Value] = None
        valid: Optional[Value] = None
        for name, stride in zip(self.base_names, self.base_strides):
            if name not in coords:
                raise ValueError(
                    f"after chain, base coord {name!r} not in {sorted(coords.keys())}"
                )
            c = coords[name]
            valid = _and(b, valid, c.valid)
            if name == base_coord:
                # i64 term -- no 2 GiB overflow. The base feeds a buffer
                # descriptor (uniform per wave), so pin the (wave-uniform)
                # block id to an SGPR before widening.
                base_val = b.to_sgpr_u32(c.value)
                base_i64 = b.mul(b.zext(base_val, I64), b.const_i64(int(stride)))
            else:
                term = c.value if stride == 1 else b.mul(c.value, b.const_i32(stride))
                within = term if within is None else b.add(within, term)
        if base_i64 is None:
            raise ValueError(
                f"offset_i64_split: base_coord {base_coord!r} not among "
                f"base coords {list(self.base_names)}"
            )
        if within is None:
            within = b.const_i32(0)
        return base_i64, within, valid

    def offset_i64(
        self,
        b: IRBuilder,
        i64_coord: str,
        **upper_values: Value,
    ) -> Tuple[Value, Optional[Value]]:
        """Like :meth:`offset` but returns the FULL offset as a single i64.

        ``i64_coord``'s term is widened to i64 (so it cannot overflow for
        paged caches > 2 GiB); the other terms are zero-extended and summed
        in i64. Unlike :meth:`offset_i64_split` this does NOT scalarise, so
        it is for **flat per-lane global loads** where the block id varies
        per lane (e.g. the fp8 sync-dequant loader). The returned i64
        offset is in the descriptor's units (bytes here)."""
        coords = self._run_chain(b, upper_values)
        off: Optional[Value] = None
        valid: Optional[Value] = None
        for name, stride in zip(self.base_names, self.base_strides):
            if name not in coords:
                raise ValueError(
                    f"after chain, base coord {name!r} not in {sorted(coords.keys())}"
                )
            c = coords[name]
            valid = _and(b, valid, c.valid)
            if name == i64_coord:
                term = b.mul(b.zext(c.value, I64), b.const_i64(int(stride)))
            else:
                t32 = c.value if stride == 1 else b.mul(c.value, b.const_i32(stride))
                term = b.zext(t32, I64)
            off = term if off is None else b.add(off, term)
        if off is None:
            off = b.const_i64(0)
        return off, valid

    def _run_chain_delta(
        self,
        b: IRBuilder,
        old_values: Dict[str, Value],
        delta_values: Dict[str, Value],
    ) -> Tuple[Dict[str, CoordVar], Dict[str, CoordVar]]:
        """Topologically lower a *delta* through the chain.

        Returns ``(delta_coords, new_coords)`` for every coord down to the
        base level. ``delta_coords[name].value`` is the i32 change of coord
        ``name``; ``new_coords[name]`` is the new absolute coord (value +
        validity). Linear transforms derive their lower deltas straight from
        the upper deltas (so a loop-invariant delta folds to a constant);
        non-linear transforms (``Unmerge*``/``Modulo``/``XorT``) recompute
        their lowers from the new absolute upper coords, exactly mirroring
        each C++ transform's ``update_lower_index``.
        """
        missing = set(self.upper_names) - set(old_values.keys())
        if missing:
            raise ValueError(
                f"move() missing upper coords for descriptor {self.name!r}: "
                f"{sorted(missing)}"
            )
        old: Dict[str, CoordVar] = {n: CoordVar(n, v) for n, v in old_values.items()}
        new: Dict[str, CoordVar] = {}
        delta: Dict[str, CoordVar] = {}
        for n in self.upper_names:
            d = delta_values.get(n)
            if d is None:
                d = b.const_i32(0)
            delta[n] = CoordVar(n, d)
            new[n] = CoordVar(n, b.add(old_values[n], d))
        remaining: List[Transform] = list(self.chain)
        while remaining:
            progress = False
            next_remaining: List[Transform] = []
            for t in remaining:
                if all(name in old for name in t.upper):
                    old_low = t.apply(b, old)
                    new_low = t.apply(b, new)
                    diff_low = t.update_lower(b, old, delta, new)
                    for n in t.lower:
                        old[n] = old_low[n]
                        new[n] = new_low[n]
                        delta[n] = CoordVar(n, diff_low[n].value)
                    progress = True
                else:
                    next_remaining.append(t)
            if not progress:
                names = [t.upper for t in next_remaining]
                avail = sorted(old.keys())
                raise ValueError(
                    f"move chain has unresolved deps: pending uppers "
                    f"= {names}; available coords = {avail}"
                )
            remaining = next_remaining
        return delta, new

    def move(
        self,
        b: IRBuilder,
        prev_offset: Value,
        old_values: Dict[str, Value],
        delta_values: Dict[str, Value],
    ) -> Tuple[Value, Optional[Value]]:
        """Incrementally update a cached offset by a coordinate *delta*.

        Mirrors CK Tile's ``move_tensor_coordinate``: instead of re-running
        the whole transform chain from scratch (as :meth:`offset` does), this
        propagates the per-coord deltas through each transform's
        ``update_lower_index`` and adds the resulting base-level offset delta
        to ``prev_offset``. For a loop-invariant ``delta_values`` (the GEMM
        K-loop / sliding-window step) the delta folds to a single constant
        add, so the K-loop body emits no div/mod and no full address recompute.

        ``prev_offset`` is the offset returned by the previous :meth:`offset`
        / :meth:`move`. ``old_values`` are the upper coords that produced it;
        ``delta_values`` maps a subset of ``upper_names`` to their i32 change
        (omitted coords are treated as unchanged). Returns the new
        ``(offset, valid)`` -- ``valid`` is recomputed at the new absolute
        position so boundary checks stay correct.
        """
        delta, new = self._run_chain_delta(b, old_values, delta_values)
        off_delta: Optional[Value] = None
        valid: Optional[Value] = None
        for name, stride in zip(self.base_names, self.base_strides):
            if name not in delta:
                raise ValueError(
                    f"after chain, base coord {name!r} not in {sorted(delta.keys())}"
                )
            valid = _and(b, valid, new[name].valid)
            dv = delta[name].value
            term = dv if stride == 1 else b.mul(dv, b.const_i32(stride))
            off_delta = term if off_delta is None else b.add(off_delta, term)
        if off_delta is None:
            return prev_offset, valid
        return b.add(prev_offset, off_delta), valid


__all__ = [
    "CoordVar",
    "Embed",
    "Freeze",
    "Indirect",
    "Insert",
    "LeftPad",
    "Merge",
    "Modulo",
    "Offset",
    "Pad",
    "PadDynamic",
    "PassThrough",
    "Replicate",
    "RightPad",
    "Slice",
    "TensorDescriptor",
    "Transform",
    "Unmerge",
    "UnmergeDivMod",
    "UnmergeMagicDiv",
    "XorT",
    "calculate_magic_numbers",
    "do_magic_division",
    "embed",
    "freeze",
    "indirect",
    "insert",
    "left_pad",
    "merge",
    "merge_magic",
    "modulo",
    "offset",
    "pad",
    "pad_dynamic",
    "pass_through",
    "replicate",
    "right_pad",
    "slice_",
    "unmerge",
    "unmerge_div_mod",
    "unmerge_magic",
    "xor_t",
]
