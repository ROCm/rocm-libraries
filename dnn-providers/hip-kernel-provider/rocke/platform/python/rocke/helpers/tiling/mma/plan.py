# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``TileMmaPlan`` -- the MMA DESIGN object (target-aware resolution + A/B/C layouts).

The author states logical intent (a WAVE-TILE shape + dtypes) and a bound target; the plan resolves
the concrete intrinsic (from the atom shape) and exposes the A/B/C layouts -- all IR-free (no builder).
It answers "what the instruction needs", never "how the data gets there or how the grid is walked"
(that is :class:`~rocke.helpers.tiling.mma.driver.TileMmaDriver`). The front-door
:class:`~rocke.helpers.tiling.mma.mma_operation.TileMma` composes a plan + a driver; an advanced author
can hold a plan directly (toolbox tier -- see ``docs/tiling_api_contract.md``).

The tile knobs live on :class:`Tiling`: ``atom_shape`` (the hardware atom; the inner-K count is
``wave_K / atom_K``) and ``order`` (M/N/K subtile iteration order). The raw encoding is never exposed as
such -- ``a_layout`` / ``b_layout`` / ``c_layout`` return :class:`WarpDistributionEncoding` values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..encoding import WarpDistributionEncoding
from ..fragments import TileDesc
from ..traits import MmaTraits, MmaTraitsCatalog, load_mma_traits
from .styles import CanonicalStyle, LayoutStyle
from .warp_encoding import a_warp_encoding, b_warp_encoding, c_warp_encoding

__all__ = ["Tiling", "TileMmaPlan"]

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
      Iterated inside one ``TileMmaDriver`` call.
    * ``mac_prio`` -- ``s_setprio`` level (``0..3``): a PERFORMANCE-TUNING knob raised for the
      matrix-dense body of the wave-tile cluster (MFMA or WMMA -- the driver is instruction-agnostic).
      The driver issues the first atom at normal priority, raises to ``mac_prio`` for the rest, and
      drops back to ``0`` after the last. Like ``order`` this reorders ISSUE, not the math (bit-exact),
      so it is a schedule knob, not a correctness one -- worth sweeping per arch/shape. ``0`` (the
      default) emits nothing, keeping the recording byte-identical; a single-atom cluster emits nothing
      regardless -- there is no dense body to protect.
    """

    atom_shape: Optional[tuple[int, int, int] | str] = None
    order: str = "MNK"
    mac_prio: int = 0

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
        if (not isinstance(self.mac_prio, int) or isinstance(self.mac_prio, bool)
                or not 0 <= self.mac_prio <= 3):
            raise ValueError(f"mac_prio must be an int in 0..3 -- mac_prio={self.mac_prio!r}")


class TileMmaPlan:
    """A target-resolved wave-tile MMA design (dense; the M1 surface). IR-free.

    ``TileMmaPlan((16, 16, 64), a="f16", b="f16", c="f32", target="gfx90a",
    tiling=Tiling(atom_shape=(16, 16, 16)))`` -- resolution + validation happen here, fail-fast, with
    no IR. Feed the resolved layouts / ``emit_op`` to a :class:`TileMmaDriver` to walk the atom grid.
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
        style: LayoutStyle | None = None,
        catalog: MmaTraitsCatalog | None = None,
        atom_override: str | None = None,
    ) -> None:
        tiling = tiling if tiling is not None else Tiling()
        style = style if style is not None else CanonicalStyle()
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
        self._style = style
        self._m_subtiles = m_sub
        self._n_subtiles = n_sub
        self._k_subtiles = k_sub
        self._a_dtype = a
        self._b_dtype = b
        self._c_dtype = c
        self._target = target

        # C-oracle (rev-7 Principle 8): the DERIVED native C the style produces must equal the machine's
        # fall-out from the style's operands. Two INDEPENDENT paths -- the accumulator descriptor's own
        # labels vs `derive_c_distribution` flowing the operands through the fixed canonical machine.
        # The two shipped styles pass; a mis-generalized future style whose C geometry the atom-derived
        # descriptor cannot express is rejected here, not silently mis-stored. C is always derived (never
        # style-supplied); a style only influences C via the K-distribution its operands present.
        # Per-operand soundness at construction (SAME timing as the C-oracle): a custom style whose
        # operand descriptor is per-operand-unsound (a wandering M/N) is rejected here at plan build,
        # not only later at TileMmaDriver.__call__. Derived / canonical / interleaved pass by construction.
        self._assert_operands_sound()
        self._assert_accumulator_matches_oracle()

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
    def mfma_count(self) -> int:
        """The number of matrix-instruction atoms the driver issues per wave-tile cluster (the
        product of the subtile grid). The single source for a schedule's MFMA count -- never re-typed
        by a kernel."""
        return self._m_subtiles * self._n_subtiles * self._k_subtiles

    @property
    def tiling(self) -> Tiling:
        return self._tiling

    @property
    def target(self) -> str:
        return self._target

    @property
    def a_dtype(self) -> str:
        return self._a_dtype

    @property
    def b_dtype(self) -> str:
        return self._b_dtype

    @property
    def c_dtype(self) -> str:
        return self._c_dtype

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

    def a_desc(self):
        """A operand `TileDesc` -- wave (M, K) shape + canonical `a_layout`, ready for `load_fragment`.
        For a non-canonical operand layout, pass ``style=`` and use :attr:`a_operand_desc`."""
        layout = a_warp_encoding(self._traits, m_iter=self._m_subtiles, k_iter=self._k_subtiles)
        return TileDesc((self._shape[0], self._shape[2]), layout)

    def b_desc(self):
        """B operand `TileDesc` -- wave (N, K) shape + canonical `b_layout` (see :meth:`a_desc`)."""
        layout = b_warp_encoding(self._traits, n_iter=self._n_subtiles, k_iter=self._k_subtiles)
        return TileDesc((self._shape[1], self._shape[2]), layout)

    @property
    def c_desc(self):
        """C accumulator `TileDesc` -- wave (M, N) shape + `c_layout` (no interleaved variant)."""
        return TileDesc((self._shape[0], self._shape[1]), self.c_layout)

    @property
    def style(self) -> LayoutStyle:
        """The resolved layout style (default :class:`CanonicalStyle`)."""
        return self._style

    @property
    def a_operand_desc(self) -> TileDesc:
        """A operand MMA-ready `TileDesc` for the whole wave tile, as the resolved STYLE produces it.
        DISTINCT from `a_layout`: canonical makes them coincide; interleaved reorders the registers.
        A fragment is built from THIS (and the driver slices it); `a_layout`/`b_layout` stay the
        atom-canonical, style-immutable `operand_soundness` machine reference."""
        return self._style.operand_desc(
            self._traits, role="A", free_sub=self._m_subtiles, k_sub=self._k_subtiles
        )

    @property
    def b_operand_desc(self) -> TileDesc:
        """B operand MMA-ready `TileDesc` for the whole wave tile, as the resolved STYLE produces it
        (see :attr:`a_operand_desc`)."""
        return self._style.operand_desc(
            self._traits, role="B", free_sub=self._n_subtiles, k_sub=self._k_subtiles
        )

    @property
    def c_native_desc(self) -> TileDesc:
        """The DERIVED native C accumulator `TileDesc` for the resolved style. Always atom-derived -- a
        style never supplies a C descriptor; it only influences C via the K-distribution its operands
        present to the atom (validated by the C-oracle at construction)."""
        return self._style.accumulator_desc(
            self._traits, m_sub=self._m_subtiles, n_sub=self._n_subtiles
        )

    def _assert_operands_sound(self) -> None:
        """Per-operand soundness at CONSTRUCTION: each operand descriptor the style produces must be a
        sound MMA operand (one fixed M/N per output-row, well-formed K) against the atom-canonical
        machine. Catches a per-operand-unsound custom style at plan build, matching the C-oracle's
        timing; the driver keeps its own unconditional check. The reference is the atom-canonical
        ``a_layout``/``b_layout`` -- NEVER the style's own descriptor (that would be vacuous)."""
        from ..transforms import operand_soundness

        for role, operand_desc, canon in (
            ("A", self.a_operand_desc, self.a_layout),
            ("B", self.b_operand_desc, self.b_layout),
        ):
            d = operand_soundness(operand_desc.layout, canon, role=role)
            if d.severity != "ok":
                raise ValueError(
                    f"style {self._style.name!r} {role} operand not sound for {self.op_id!r} -- {d.message}"
                )

    def _assert_accumulator_matches_oracle(self) -> None:
        """Independent-path C-oracle: the style's native accumulator labels vs the machine's fall-out
        from the style's operands (:func:`derive_c_distribution`). See the `__init__` note."""
        from ..transforms import derive_c_distribution
        from ..transforms._core import as_forward_map

        native = as_forward_map(self.c_native_desc.layout)
        oracle = derive_c_distribution(
            self.a_operand_desc.layout,
            self.b_operand_desc.layout,
            a_canon=self.a_layout,
            b_canon=self.b_layout,
            c_canon=self.c_layout,
        )
        if native != oracle:
            raise ValueError(
                f"style {self._style.name!r} accumulator is inconsistent with the machine for "
                f"{self.op_id!r}: its native C labels differ from `derive_c_distribution` flowing the "
                f"style's operands through the atom. C is always DERIVED -- a style whose C geometry the "
                f"atom-derived descriptor cannot express is not supported."
            )

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

    def __repr__(self) -> str:
        return (
            f"TileMmaPlan(shape={self._shape}, atom={self._atom_shape}, "
            f"a={self._a_dtype!r}, b={self._b_dtype!r}, c={self._c_dtype!r}, "
            f"target={self._target!r}, op_id={self.op_id!r})"
        )
