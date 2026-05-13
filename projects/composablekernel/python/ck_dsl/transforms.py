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
from typing import Dict, List, Optional, Sequence, Tuple

from .core.ir import F16, IRBuilder, Type, Value


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

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        return {self.lower[0]: replace(u, name=self.lower[0])}


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

    def apply(self, b: IRBuilder, coords: Dict[str, CoordVar]) -> Dict[str, CoordVar]:
        u = coords[self.upper[0]]
        c_lo = b.const_i32(self.lo)
        c_hi = b.const_i32(self.hi)
        valid = _and(b, _ge(b, u.value, c_lo), _lt(b, u.value, c_hi))
        merged_valid = _and(b, u.valid, valid)
        return {self.lower[0]: CoordVar(self.lower[0], u.value, merged_valid)}


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

        Implementation: we run the chain in topological order. A
        transform T can be applied once all of `T.upper` coords are
        in `coords`. We repeat until no more transforms are
        applicable; if any transforms remain unapplied at that point
        the chain has an unresolvable dependency cycle or a typo in
        coord names (we raise with a precise diagnostic).
        """
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
                    # Don't actually remove uppers from `coords`: an
                    # intermediate coord may be needed by multiple
                    # transforms (e.g. `c` appears in two unmerges).
                    # The chain-validity invariant (computed in
                    # transform()) guarantees the user-facing coords
                    # are exactly those whose final value the
                    # descriptor consumes.
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
        # Now coords should contain exactly self.base_names.
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


__all__ = [
    "CoordVar",
    "Embed",
    "Merge",
    "Pad",
    "PassThrough",
    "TensorDescriptor",
    "Transform",
    "Unmerge",
    "embed",
    "merge",
    "pad",
    "pass_through",
    "unmerge",
]
