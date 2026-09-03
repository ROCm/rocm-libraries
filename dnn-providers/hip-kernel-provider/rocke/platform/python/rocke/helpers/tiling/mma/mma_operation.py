# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public MMA operation -- target-aware resolution from logical intent.

``TileMma`` is the author-facing MMA object. The author states logical intent (a WAVE-TILE
shape + dtypes) and a bound target; ``TileMma`` resolves the concrete intrinsic (from the
atom shape) and exposes the A/B/C layouts -- all **IR-free** (no builder). The wave tile
may be a multiple of the atom: the object then OWNS the subtile iteration -- one
``mma(b, a, b, acc)`` call walks the atom grid internally (M1: the K subtiles), so the
author never hand-loops atoms.

The tile knobs live on the object, folded into a :class:`Tiling` policy: ``atom_shape``
(the hardware atom; the inner-K count is ``wave_K / atom_K``) and ``order`` (M/N subtile
iteration order). The raw encoding is never exposed as such -- ``a_layout`` / ``b_layout``
/ ``c_layout`` return :class:`WarpDistributionEncoding` values the layout system /
reflection consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..encoding import WarpDistributionEncoding
from ..fragments import Fragment, TileDesc, fragment_length
from ..traits import MmaTraits, MmaTraitsCatalog, load_mma_traits
from ..transforms import validate_operands
from .warp_encoding import a_warp_encoding, b_warp_encoding, c_warp_encoding

__all__ = ["Tiling", "TileMma"]

# The subtile loop-nest order is any permutation of the three axes. Consistent with STRIDE
# notation, the RIGHT-MOST axis is the fastest-varying (innermost) loop and the left-most is
# the slowest (outermost) -- so "MNK" iterates K innermost, like strides (..., 1). C
# accumulation is commutative, so every order yields the same (bit-exact) result -- the knob
# is for schedule/locality, not correctness.
_SUBTILE_ORDERS = ("MNK", "MKN", "NMK", "NKM", "KMN", "KNM")

@dataclass(frozen=True)
class Tiling:
    """The wave-tile iteration policy -- the MMA object's knobs.

    * ``atom_shape`` -- how to pick the hardware MMA atom. Two flavours (plus a default):
        - a shape ``(M, N, K)`` tuple -> TARGET-AGNOSTIC: resolved to the atom for the bound
          target + dtypes (MFMA on CDNA, WMMA on RDNA -- authorship carries no gfx branch);
        - an explicit intrinsic name ``str`` (e.g. ``"mfma_f32_16x16x16f16"``) -> the escape
          hatch: that exact backend op is used (target-specific by construction);
        - ``None`` -> the wave shape itself (a single MMA).
      The inner-K count the object runs is ``wave_K / atom_K``.
    * ``order`` -- the M/N/K subtile loop-nest order (a permutation of ``"MNK"``). Stride
      convention: the RIGHT-MOST axis varies fastest (innermost); ``"MNK"`` runs K innermost.
      Iterated inside one ``mma(...)`` call.
    """

    atom_shape: Optional[tuple[int, int, int] | str] = None
    order: str = "MNK"

    def __post_init__(self) -> None:
        atom = self.atom_shape
        if atom is not None and not isinstance(atom, str):
            if len(atom) != 3 or any(not isinstance(d, int) or d <= 0 for d in atom):
                raise ValueError(
                    f"atom_shape must be 3 positive ints or an intrinsic name -- "
                    f"atom_shape={atom!r}"
                )
        if self.order not in _SUBTILE_ORDERS:
            raise ValueError(
                f"unknown subtile order -- order={self.order!r}, "
                f"expected one of {list(_SUBTILE_ORDERS)}"
            )

class TileMma:
    """A target-resolved wave-tile MMA operation + subtile driver (dense; the M1 surface).

    Author call: ``TileMma((16, 16, 64), a="f16", b="f16", c="f32", target="gfx90a",
    tiling=Tiling(atom_shape=(16, 16, 16)))``. Resolution + validation happen here,
    fail-fast, with no IR. Calling the instance (``mma(b, a_frag, b_frag, acc)``) walks the
    atom grid internally and returns the updated accumulator `Fragment`.
    """

    def __init__(
        self,
        shape: tuple[int, int, int] | None = None,
        *,
        a: str | None = None,
        b: str | None = None,
        c: str | None = None,
        target: str,
        tiling: Tiling | None = None,
        catalog: MmaTraitsCatalog | None = None,
        atom_override: str | None = None,
    ) -> None:
        tiling = tiling if tiling is not None else Tiling()
        catalog = catalog if catalog is not None else load_mma_traits()

        self._traits: MmaTraits
        if atom_override is not None:
            # Name a specific intrinsic and DERIVE its M/N/K + dtypes from the traits (target still
            # required, to validate/resolve). `shape`, if given, is the wave tile (a multiple of the
            # atom); otherwise it IS the atom -- a single MMA. Any a/b/c passed are overridden.
            if tiling.atom_shape is not None:
                raise ValueError(
                    "specify the atom via atom_override OR tiling.atom_shape, not both"
                )
            self._traits = self._resolve_atom(catalog, atom_override, target)
            atom_shape = (self._traits.m, self._traits.n, self._traits.k)
            a = b = self._traits.input_dtype
            c = self._traits.output_dtype
            shape = tuple(shape) if shape is not None else atom_shape
        else:
            if shape is None:
                raise ValueError("shape (M, N, K) is required unless atom_override is given")
            if a is None or b is None or c is None:
                raise ValueError("a/b/c dtypes are required unless atom_override is given")
            shape = tuple(shape)
            if a != b:
                raise ValueError(f"MFMA requires matching A/B dtypes -- a={a!r}, b={b!r}")
            # Resolve the atom + its traits from the knob: explicit intrinsic NAME, atom SHAPE,
            # or (None) the wave shape itself. Both paths end with a resolved traits row.
            atom_knob = tiling.atom_shape
            if isinstance(atom_knob, str):
                self._traits = self._resolve_by_name(catalog, atom_knob, target, a, c)
                atom_shape = (self._traits.m, self._traits.n, self._traits.k)
            else:
                atom_shape = atom_knob if atom_knob is not None else shape
                m, n, k = atom_shape
                self._traits = catalog.select(
                    target=target, input_dtype=a, output_dtype=c, m=m, n=n, k=k, family="dense"
                )

        if len(shape) != 3:
            raise ValueError(f"shape must be (M, N, K) -- got {shape!r}")

        # Wave tile must be an integer multiple of the atom on every axis.
        subtiles = []
        for axis, wave_dim, atom_dim in zip("MNK", shape, atom_shape):
            if wave_dim % atom_dim != 0:
                raise ValueError(
                    f"wave {axis} ({wave_dim}) is not an integer multiple of atom "
                    f"{axis} ({atom_dim})"
                )
            subtiles.append(wave_dim // atom_dim)
        m_sub, n_sub, k_sub = subtiles

        self._shape = shape
        self._atom_shape = atom_shape
        self._tiling = tiling
        self._m_subtiles = m_sub
        self._n_subtiles = n_sub
        self._k_subtiles = k_sub
        self._a_dtype = a
        self._b_dtype = b
        self._c_dtype = c
        self._target = target

    @staticmethod
    def _resolve_by_name(
        catalog: MmaTraitsCatalog, op_id: str, target: str, a: str, c: str
    ) -> MmaTraits:
        """Resolve an EXPLICIT intrinsic name (escape hatch) to its traits, validating that it
        exists, runs on ``target``, and matches the requested dtypes. ``catalog.get`` already
        fails fast for unknown/reserved op_ids."""
        traits = catalog.get(op_id)
        if not traits.supports(target):
            raise ValueError(
                f"MMA intrinsic not available on target -- op_id={op_id!r}, "
                f"target={target!r}, supported={list(traits.supported_targets)}"
            )
        if traits.input_dtype != a or traits.output_dtype != c:
            raise ValueError(
                f"MMA intrinsic dtype mismatch -- op_id={op_id!r}, intrinsic "
                f"in/out=({traits.input_dtype},{traits.output_dtype}), requested ({a},{c})"
            )
        return traits

    @staticmethod
    def _resolve_atom(catalog: MmaTraitsCatalog, op_id: str, target: str) -> MmaTraits:
        """Resolve an ``atom_override`` intrinsic NAME to its traits: validate it exists and runs
        on ``target``. M/N/K + dtypes are DERIVED from it, so there is no dtype-match check (unlike
        ``_resolve_by_name``, which validates against author-supplied dtypes)."""
        traits = catalog.get(op_id)
        if not traits.supports(target):
            raise ValueError(
                f"MMA intrinsic not available on target -- op_id={op_id!r}, "
                f"target={target!r}, supported={list(traits.supported_targets)}"
            )
        return traits

    @property
    def shape(self) -> tuple[int, int, int]:
        """The wave-tile (M, N, K) the object drives (a multiple of the atom)."""
        return self._shape

    @property
    def atom_shape(self) -> tuple[int, int, int]:
        """The hardware MMA atom (M, N, K) the intrinsic resolved to."""
        return self._atom_shape

    @property
    def subtiles(self) -> tuple[int, int, int]:
        """The (M, N, K) atom-grid the object iterates = wave shape / atom shape."""
        return (self._m_subtiles, self._n_subtiles, self._k_subtiles)

    @property
    def tiling(self) -> Tiling:
        return self._tiling

    @property
    def target(self) -> str:
        return self._target

    @property
    def op_id(self) -> str:
        """The concrete intrinsic the target resolved to (e.g. mfma_f32_16x16x16f16)."""
        return self._traits.op_id

    @property
    def wave_size(self) -> int:
        return self._traits.wave_size

    @property
    def traits(self) -> MmaTraits:
        """The resolved traits (for reflection)."""
        return self._traits

    @property
    def a_layout(self) -> WarpDistributionEncoding:
        """A operand layout for the WHOLE wave tile (M and K subtiles folded in)."""
        return a_warp_encoding(
            self._traits, m_iter=self._m_subtiles, k_iter=self._k_subtiles
        )

    @property
    def b_layout(self) -> WarpDistributionEncoding:
        """B operand layout for the WHOLE wave tile (N and K subtiles folded in)."""
        return b_warp_encoding(
            self._traits, n_iter=self._n_subtiles, k_iter=self._k_subtiles
        )

    @property
    def c_layout(self) -> WarpDistributionEncoding:
        """C accumulator layout for the WHOLE wave tile (M and N subtiles; K contracted)."""
        return c_warp_encoding(
            self._traits, m_iter=self._m_subtiles, n_iter=self._n_subtiles
        )

    def a_desc(self, *, interleaved: bool = False):
        """A operand `TileDesc` -- wave (M, K) shape + canonical `a_layout`, ready for `load_fragment`.

        ``interleaved=True`` is BROKEN and raises: it does NOT produce a proper interleaved layout.
        Build interleaved layouts as custom static tile distributions (`make_tile_desc`)."""
        layout = a_warp_encoding(
            self._traits, m_iter=self._m_subtiles, k_iter=self._k_subtiles, interleaved=interleaved
        )
        return TileDesc((self._shape[0], self._shape[2]), layout)

    def b_desc(self, *, interleaved: bool = False):
        """B operand `TileDesc` -- wave (N, K) shape + canonical `b_layout`.

        ``interleaved=True`` is BROKEN and raises (see :meth:`a_desc`)."""
        layout = b_warp_encoding(
            self._traits, n_iter=self._n_subtiles, k_iter=self._k_subtiles, interleaved=interleaved
        )
        return TileDesc((self._shape[1], self._shape[2]), layout)

    @property
    def c_desc(self):
        """C accumulator `TileDesc` -- wave (M, N) shape + `c_layout` (no interleaved variant)."""
        return TileDesc((self._shape[0], self._shape[1]), self.c_layout)

    # SOT dtype token -> arch-catalog token (naming-convention alias only).
    _ARCH_DTYPE_ALIAS = {"f16": "fp16", "bf16": "bf16", "f32": "fp32", "f64": "fp64"}

    def emit_op(self):
        """Resolve the backend ``MmaOp`` for ``b.mma`` from the arch op-registry -- keyed by
        the ATOM shape (one ``b.mma`` consumes one atom).

        Lazy import: uses ``core.arch`` -- the op registry, NOT the mfma emission helpers --
        so this layer stays independent of ``mfma_gemm_inner``.
        """
        from rocke.core.arch import ArchTarget

        target = ArchTarget.from_gfx(self._target)
        m, n, k = self._atom_shape
        op = target.mma.op_for_shape(
            family="mma",
            a_dtype=self._ARCH_DTYPE_ALIAS.get(self._a_dtype, self._a_dtype),
            b_dtype=self._ARCH_DTYPE_ALIAS.get(self._b_dtype, self._b_dtype),
            c_dtype=self._ARCH_DTYPE_ALIAS.get(self._c_dtype, self._c_dtype),
            m=m,
            n=n,
            k=k,
        )
        if op is None:
            raise ValueError(
                f"no backend MMA op for atom={self._atom_shape} "
                f"{self._a_dtype}->{self._c_dtype} on target={self._target!r}"
            )
        return op

    def _ir_type(self, token: str):
        """SOT dtype token (``f16``/``f32``/...) -> rocke ``ir.Type`` (lazy import seam)."""
        from rocke.core import ir

        try:
            return getattr(ir, {"f16": "F16", "bf16": "BF16", "f32": "F32",
                                 "f8": "FP8E4M3", "bf8": "BF8E5M2"}[token])
        except KeyError as exc:
            raise NotImplementedError(
                f"no ir.Type for dtype token -- token={token!r}"
            ) from exc

    @staticmethod
    def _read_subvector(b, vec, start: int, length: int, dtype):
        """Extract one atom's contiguous register slice ``[start:start+length]`` into a fresh
        ``<length x dtype>`` vector for ``b.mma``."""
        out = b.zero_vec(dtype, length)
        for i in range(length):
            out = b.vec_insert(out, b.vec_extract(vec, start + i), i)
        return out

    @staticmethod
    def _write_subvector(b, vec, sub, start: int, length: int):
        """Write ``sub`` back into ``vec`` at ``[start:start+length]``, returning the new SSA
        vector (accumulators are loop-carried SSA values, so this rebuilds the tile C)."""
        out = vec
        for i in range(length):
            out = b.vec_insert(out, b.vec_extract(sub, i), start + i)
        return out

    def _subtile_triples(self):
        """The (mi, nj, ki) atom visitation order, per ``tiling.order`` (right-most fastest)."""
        ranges = {
            "M": range(self._m_subtiles),
            "N": range(self._n_subtiles),
            "K": range(self._k_subtiles),
        }
        order = self._tiling.order
        triples = []
        for x0 in ranges[order[0]]:
            for x1 in ranges[order[1]]:
                for x2 in ranges[order[2]]:
                    axis = {order[0]: x0, order[1]: x1, order[2]: x2}
                    triples.append((axis["M"], axis["N"], axis["K"]))
        return triples

    def __call__(self, b, a_fragment, b_fragment, accumulator):
        """Walk the M x N x K atom grid for the wave tile (in ``tiling.order``), issuing one
        ``b.mma`` per atom and accumulating each C subtile. The fragments are
        subtile-contiguous (from the wave layouts), so every atom is a register slice.
        Validates operand dtypes AND K-alignment first."""
        for name, fragment in (("A", a_fragment), ("B", b_fragment), ("C", accumulator)):
            want = self._ir_type({"A": self._a_dtype, "B": self._b_dtype,
                                  "C": self._c_dtype}[name])
            if fragment.dtype.name != want.name:
                raise ValueError(
                    f"MMA operand dtype mismatch -- operand={name}, "
                    f"fragment={fragment.dtype.name!r}, expected {want.name!r}"
                )

        # MMA safety (pairwise half of the sound MAC; correctness SOT: docs/mma_is_machinery.md): the
        # hardware pairs A-slot-s with B-slot-s and sums over K, so A and B must share the same positional
        # K-distribution (M/N register order is free -- you choose the constant; K order need not be
        # canonical). Per-operand soundness (M/N fixed per output) holds by construction here (fragments
        # are atom register-reorders). A mismatched pair is rejected with a fix hint.
        # Validate K PER ATOM: the driver pairs (mi,ki)*(nj,ki), so A's m_sub M-atoms and B's n_sub
        # N-atoms each only need their atom-K to match -- comparing the whole (multi-atom) fragments
        # would falsely reject rectangular wave tiles where m_sub != n_sub (register counts differ).
        ok, why = validate_operands(
            a_fragment.tile_desc.layout, b_fragment.tile_desc.layout,
            a_free_atoms=self._m_subtiles, b_free_atoms=self._n_subtiles,
        )
        if not ok:
            raise ValueError(f"MMA operands not K-aligned for {self.op_id!r} -- {why}")

        op = self.emit_op()
        m_sub, n_sub, k_sub = self._m_subtiles, self._n_subtiles, self._k_subtiles

        # Single C subtile: accumulate in-register over K (byte-identical to the atom path).
        if m_sub == 1 and n_sub == 1:
            acc_value = accumulator.value
            if k_sub == 1:
                return Fragment(
                    accumulator.tile_desc, accumulator.dtype,
                    b.mma(op, a_fragment.value, b_fragment.value, acc_value),
                )
            a_atom = fragment_length(a_fragment.tile_desc.layout) // k_sub
            b_atom = fragment_length(b_fragment.tile_desc.layout) // k_sub
            for ki in range(k_sub):
                a_sub = self._read_subvector(b, a_fragment.value, ki * a_atom, a_atom, a_fragment.dtype)
                b_sub = self._read_subvector(b, b_fragment.value, ki * b_atom, b_atom, b_fragment.dtype)
                acc_value = b.mma(op, a_sub, b_sub, acc_value)
            return Fragment(accumulator.tile_desc, accumulator.dtype, acc_value)

        # Subtiled M/N grid. Carry a PER-ATOM accumulator SSA for each (mi, nj) C subtile so
        # that across K every C subtile is touched ONLY by `b.mma` (an MFMA def->use chain) --
        # no `vec_extract`/`vec_insert` on C inside the K-loop. LLVM then keeps each atom's
        # accumulator in an AGPR (the MFMA writes acc natively and reads Cin from acc), instead
        # of spilling the whole C tile into arch VGPRs (which a monolithic extract/insert-per-K
        # forces). The incoming C is split into per-atom SSAs ONCE (prologue) and packed back
        # ONCE (epilogue), off the K-loop. Any loop-nest order is correct (C accum is commutative).
        a_atom = fragment_length(a_fragment.tile_desc.layout) // (m_sub * k_sub)
        b_atom = fragment_length(b_fragment.tile_desc.layout) // (n_sub * k_sub)
        c_atom = fragment_length(accumulator.tile_desc.layout) // (m_sub * n_sub)
        accs = [
            self._read_subvector(b, accumulator.value, idx * c_atom, c_atom, accumulator.dtype)
            for idx in range(m_sub * n_sub)
        ]
        for mi, nj, ki in self._subtile_triples():
            idx = mi * n_sub + nj
            a_sub = self._read_subvector(
                b, a_fragment.value, (mi * k_sub + ki) * a_atom, a_atom, a_fragment.dtype
            )
            b_sub = self._read_subvector(
                b, b_fragment.value, (nj * k_sub + ki) * b_atom, b_atom, b_fragment.dtype
            )
            accs[idx] = b.mma(op, a_sub, b_sub, accs[idx])
        result = accumulator.value
        for idx in range(m_sub * n_sub):
            result = self._write_subvector(b, result, accs[idx], idx * c_atom, c_atom)
        return Fragment(accumulator.tile_desc, accumulator.dtype, result)

    def __repr__(self) -> str:
        return (
            f"TileMma(shape={self._shape}, atom={self._atom_shape}, "
            f"a={self._a_dtype!r}, b={self._b_dtype!r}, c={self._c_dtype!r}, "
            f"target={self._target!r}, op_id={self.op_id!r})"
        )
