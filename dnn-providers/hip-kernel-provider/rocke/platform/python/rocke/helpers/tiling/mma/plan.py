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

from ....core.dtypes import normalize_dtype
from ....core.ir import Type, dtype_to_ir_type
from ..encoding import WarpDistributionEncoding
from ..fragments import TileDesc, _fragment_dtype_ok
from ..traits import MmaTraits, MmaTraitsCatalog, load_mma_traits
from .styles import CanonicalStyle, LayoutStyle
from .warp_encoding import canonical_layouts

__all__ = ["Tiling", "TileMmaPlan"]

# The subtile loop-nest order is any permutation of the three axes. Consistent with STRIDE
# notation, the RIGHT-MOST axis is the fastest-varying (innermost) loop and the left-most is
# the slowest (outermost) -- so "MNK" iterates K innermost, like strides (..., 1). C
# accumulation is commutative, so every order yields the same (bit-exact) result -- the knob
# is for schedule/locality, not correctness.
_SUBTILE_ORDERS = ("MNK", "MKN", "NMK", "NKM", "KMN", "KNM")


def _ir_constant_name(dtype: Type) -> str:
    """The ``rocke.core.ir`` constant an author writes for ``dtype`` (``F16`` for ``f16``): every
    scalar IR type is exported under its name in upper case."""
    return f"rocke.core.ir.{dtype.name.upper()}"


def _require_scalar_ir_type(role: str, value: object) -> Type:
    """Return ``value`` when it is a scalar IR type; otherwise raise naming the constant to pass.

    ``role`` is the keyword the value came in (``"a"``, ``"b"``, ``"c"`` or ``"d"``). A dtype string
    is refused rather than converted, so every MMA dtype the plan holds is the IR type the author
    wrote.
    """
    ok, _ = _fragment_dtype_ok(value)
    if ok:
        return value  # type: ignore[return-value]
    head = f"{role}= must be a scalar IR type such as rocke.core.ir.F16"
    if isinstance(value, str):
        try:
            dtype = dtype_to_ir_type(value)
        except ValueError:
            raise ValueError(
                f"{head}, not the string {value!r} -- rocke.core.ir has no type for "
                f"{value!r}"
            ) from None
        raise ValueError(  # noqa: TRY004 -- every refused dtype argument is a ValueError
            f"{head}, not the string {value!r} -- pass {_ir_constant_name(dtype)}"
        )
    # A composite IR type (vector, pointer, LDS) is shown by its IR spelling, anything else by repr.
    shown = value.name if isinstance(value, Type) else repr(value)
    raise ValueError(f"{head} -- got {shown} ({type(value).__name__})")


@dataclass(frozen=True)
class Tiling:
    """The wave-tile iteration policy -- the MMA object's knobs.

    * ``atom_shape`` -- the hardware MMA atom's ``(M, N, K)``, a shape hint only: it is resolved
      to the instruction for the bound target + dtypes (MFMA on CDNA, WMMA on RDNA -- authorship
      carries no gfx branch). ``None`` (the default) means the wave shape itself (a single MMA).
      The inner-K count the object runs is ``wave_K / atom_K``. To pin one exact instruction, name
      it with ``TileMma(atom_override=...)`` instead.
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

    atom_shape: tuple[int, int, int] | None = None
    order: str = "MNK"
    mac_prio: int = 0

    def __post_init__(self) -> None:
        atom = self.atom_shape
        if isinstance(atom, str):
            raise ValueError(  # noqa: TRY004 -- an atom_shape the author mistyped is a ValueError
                f"atom_shape is the instruction's (M, N, K) -- got the string {atom!r}; pin an "
                f"instruction with TileMma(atom_override={atom!r})"
            )
        if atom is not None and (
            len(atom) != 3 or any(not isinstance(d, int) or d <= 0 for d in atom)
        ):
            raise ValueError(
                f"atom_shape must be 3 positive ints -- atom_shape={atom!r}"
            )
        if self.order not in _SUBTILE_ORDERS:
            raise ValueError(
                f"unknown subtile order -- order={self.order!r}, "
                f"expected one of {list(_SUBTILE_ORDERS)}"
            )
        if (
            not isinstance(self.mac_prio, int)
            or isinstance(self.mac_prio, bool)
            or not 0 <= self.mac_prio <= 3
        ):
            raise ValueError(
                f"mac_prio must be an int in 0..3 -- mac_prio={self.mac_prio!r}"
            )


class TileMmaPlan:
    """A target-resolved wave-tile MMA design (dense; the M1 surface). IR-free.

    ``TileMmaPlan((16, 16, 64), a=F16, b=F16, c=F32, target="gfx90a",
    tiling=Tiling(atom_shape=(16, 16, 16)))`` -- resolution + validation happen here, fail-fast, with
    no IR. Feed the resolved layouts / ``emit_op`` to a :class:`TileMmaDriver` to walk the atom grid.

    ``a``, ``b``, ``c`` and ``d`` are scalar IR types from ``rocke.core.ir`` (``F16``, ``F32``,
    ``FP8E4M3``, ...): A and B are the operand dtypes, ``c`` the accumulator taken in and ``d`` the
    result returned. ``d`` omitted means the result has the accumulator's dtype (``d = c``), which
    is what every accumulate-in-place instruction does. A dtype string is refused with the constant
    to pass instead. A and B may differ (``a=FP8E4M3, b=BF8E5M2``) wherever the target has such an
    instruction.

    ``atom_override="<op_id>"`` pins one exact instruction by name; ``a``/``b``/``c``/``d`` are then
    optional: each one given must be one the instruction takes, and the rest are taken from it.
    """

    def __init__(
        self,
        shape: tuple[int, int, int] | None = None,
        *,
        a: Type | None = None,
        b: Type | None = None,
        c: Type | None = None,
        d: Type | None = None,
        target: str,
        tiling: Tiling | None = None,
        style: LayoutStyle | None = None,
        catalog: MmaTraitsCatalog | None = None,
        atom_override: str | None = None,
    ) -> None:
        for role, value in (("a", a), ("b", b), ("c", c), ("d", d)):
            if value is not None:
                _require_scalar_ir_type(role, value)
        tiling = tiling if tiling is not None else Tiling()
        style = style if style is not None else CanonicalStyle()
        catalog = catalog if catalog is not None else load_mma_traits()

        self._traits: MmaTraits
        if atom_override is not None:
            # Pin one instruction by name. Its M/N/K come from the row; so do any of a/b/c/d the
            # author leaves out, and each one given must be one the row takes. `shape`, if given, is
            # the wave tile (a multiple of the atom); otherwise it IS the atom -- a single MMA.
            if tiling.atom_shape is not None:
                raise ValueError(
                    "specify the atom via atom_override OR tiling.atom_shape, not both"
                )
            self._traits, (a, b, c, d) = self._resolve_named(
                catalog, atom_override, target, a, b, c, d
            )
            atom_shape = (self._traits.m, self._traits.n, self._traits.k)
            shape = tuple(shape) if shape is not None else atom_shape
        else:
            if shape is None:
                raise ValueError(
                    "shape (M, N, K) is required unless atom_override is given"
                )
            if a is None or b is None or c is None:
                raise ValueError(
                    "a/b/c dtypes are required unless the instruction is pinned with "
                    "atom_override=..."
                )
            shape = tuple(shape)
            # The atom is the tiling knob's (M, N, K), or (None) the wave shape itself.
            atom_shape = tiling.atom_shape if tiling.atom_shape is not None else shape
            m, n, k = atom_shape
            self._traits = catalog.select(
                target=target,
                a=a,
                b=b,
                c=c,
                d=d,
                m=m,
                n=n,
                k=k,
                family="dense",
            )
            d = d if d is not None else c
        self._refuse_result_dtype_change(self._traits)

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
        self._d_dtype = d
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
    def _refuse_result_dtype_change(traits: MmaTraits) -> None:
        """Refuse a row whose result dtype (D) differs from its accumulator dtype (C).

        The plan and the driver give the backend only the accumulator dtype, and each
        instruction's result feeds the next one as its accumulator, so an instruction that returns
        a different dtype cannot be driven here. Both paths that resolve a row (selection by
        intent and ``atom_override``) end in this check.
        """
        if traits.d_dtype != traits.c_dtype:
            raise ValueError(
                f"MMA instruction {traits.op_id!r} accumulates in {traits.c_dtype} but returns "
                f"{traits.d_dtype} -- the TileMma driver does not drive an "
                f"instruction whose result dtype differs from its accumulator dtype; choose one "
                f"whose result dtype is {traits.c_dtype}"
            )

    @staticmethod
    def _resolve_named(
        catalog: MmaTraitsCatalog,
        op_id: str,
        target: str,
        a: Type | None,
        b: Type | None,
        c: Type | None,
        d: Type | None,
    ) -> tuple[MmaTraits, tuple[Type, Type, Type, Type]]:
        """Resolve an ``atom_override`` instruction name to its row and the plan's (A, B, C, D).

        The row must exist (``catalog.get`` fails fast listing the valid names) and run on
        ``target``. Each dtype the author gives must be one the row takes -- for an integer row
        whose signedness is selectable, that includes the unsigned twin, which is kept as given --
        and every dtype left out is taken from the row. An omitted ``d`` is the row's result
        dtype, never defaulted to ``c``.
        """
        traits = catalog.get(op_id)
        if not traits.supports(target):
            raise ValueError(
                f"MMA instruction not available on target -- op_id={op_id!r}, "
                f"target={target!r}, supported={list(traits.supported_targets)}"
            )
        row = dict(zip("abcd", traits.ir_dtypes()))
        row_names = {
            "a": traits.a_dtype,
            "b": traits.b_dtype,
            "c": traits.c_dtype,
            "d": traits.d_dtype,
        }
        given = {"a": a, "b": b, "c": c, "d": d}
        clauses, fixes = [], []
        for role, dtype in given.items():
            if dtype is None:
                continue
            # The catalog spells dtypes by their canonical core.dtypes names: that is the match key.
            asked = {**row_names, role: normalize_dtype(dtype.name)}
            if not traits.accepts(**asked):
                verb = "returns" if role == "d" else "takes"
                clauses.append(
                    f"{role}={dtype.name} but it {verb} {role.upper()}={row[role].name}"
                )
                fixes.append(f"{role}={_ir_constant_name(row[role])}")
        if clauses:
            omit = "a/b/c/d" if given["d"] is not None else "a/b/c"
            raise ValueError(
                f"MMA instruction dtype mismatch for {op_id!r} -- {'; '.join(clauses)} -- "
                f"pass {', '.join(fixes)}, or omit {omit} to take them from the instruction"
            )
        resolved = tuple(
            given[role] if given[role] is not None else row[role] for role in "abcd"
        )
        return traits, resolved  # type: ignore[return-value]

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
    def a_dtype(self) -> Type:
        """The A operand dtype (an IR type)."""
        return self._a_dtype

    @property
    def b_dtype(self) -> Type:
        """The B operand dtype (an IR type)."""
        return self._b_dtype

    @property
    def c_dtype(self) -> Type:
        """The accumulator dtype the instruction takes in (an IR type)."""
        return self._c_dtype

    @property
    def d_dtype(self) -> Type:
        """The result dtype the instruction returns (an IR type); equal to :attr:`c_dtype` for
        every accumulate-in-place instruction."""
        return self._d_dtype

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
    def a_desc(self) -> TileDesc:
        """A operand MMA-ready `TileDesc` for the whole wave tile, in the CONFIGURED style. The object
        returns what the author set: canonical by default, or the style's reordered operand when a
        ``style=`` was given. The canonical machine reference is NOT on the object -- the soundness
        checks build it from the ``canonical_layouts`` helper."""
        return self._style.operand_desc(
            self._traits, role="A", free_sub=self._m_subtiles, k_sub=self._k_subtiles
        )

    @property
    def b_desc(self) -> TileDesc:
        """B operand MMA-ready `TileDesc` for the whole wave tile, in the CONFIGURED style (see
        :attr:`a_desc`)."""
        return self._style.operand_desc(
            self._traits, role="B", free_sub=self._n_subtiles, k_sub=self._k_subtiles
        )

    @property
    def c_desc(self) -> TileDesc:
        """The DERIVED native C accumulator `TileDesc` for the CONFIGURED style. Always atom-derived -- a
        style never supplies a C descriptor; it only influences C via the K-distribution its operands
        present to the atom (validated by the C-oracle at construction)."""
        return self._style.accumulator_desc(
            self._traits, m_sub=self._m_subtiles, n_sub=self._n_subtiles
        )

    @property
    def a_layout(self) -> WarpDistributionEncoding:
        """A operand layout for the WHOLE wave tile, in the configured style (``a_desc.layout``)."""
        return self.a_desc.layout

    @property
    def b_layout(self) -> WarpDistributionEncoding:
        """B operand layout for the WHOLE wave tile, in the configured style (``b_desc.layout``)."""
        return self.b_desc.layout

    @property
    def c_layout(self) -> WarpDistributionEncoding:
        """C accumulator layout for the WHOLE wave tile, in the configured style (``c_desc.layout``)."""
        return self.c_desc.layout

    @property
    def style(self) -> LayoutStyle:
        """The resolved layout style (default :class:`CanonicalStyle`)."""
        return self._style

    def _assert_operands_sound(self) -> None:
        """Per-operand soundness at CONSTRUCTION: each operand descriptor the style produces must be a
        sound MMA operand (one fixed M/N per output-row, well-formed K) against the atom-canonical
        machine. Catches a per-operand-unsound custom style at plan build, matching the C-oracle's
        timing; the driver keeps its own unconditional check. The reference is the shared
        ``canonical_layouts`` helper -- NEVER the style's own descriptor (a check against itself proves nothing).
        """
        from ..transforms import mma_operand_layout_sound

        a_canon, b_canon, _ = canonical_layouts(self._traits, self.subtiles)
        for role, operand_desc, canon in (
            ("A", self.a_desc, a_canon),
            ("B", self.b_desc, b_canon),
        ):
            d = mma_operand_layout_sound(operand_desc.layout, canon, role=role)
            if d.severity != "ok":
                raise ValueError(
                    f"style {self._style.name!r} {role} operand not sound for {self.op_id!r} -- {d.message}"
                )

    def _assert_accumulator_matches_oracle(self) -> None:
        """Independent-path C-oracle: the style's native accumulator labels vs the machine's fall-out
        from the style's operands (:func:`derive_c_distribution`). See the `__init__` note.
        """
        from ..transforms import derive_c_distribution
        from ..transforms._core import as_forward_map

        a_canon, b_canon, c_canon = canonical_layouts(self._traits, self.subtiles)
        native = as_forward_map(self.c_desc.layout)
        oracle = derive_c_distribution(
            self.a_desc.layout,
            self.b_desc.layout,
            a_canon=a_canon,
            b_canon=b_canon,
            c_canon=c_canon,
        )
        if native != oracle:
            raise ValueError(
                f"style {self._style.name!r} accumulator is inconsistent with the machine for "
                f"{self.op_id!r}: its native C labels differ from `derive_c_distribution` flowing the "
                f"style's operands through the atom. C is always DERIVED -- a style whose C geometry the "
                f"atom-derived descriptor cannot express is not supported."
            )

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
            a_dtype=self._traits.a_dtype,
            b_dtype=self._traits.b_dtype,
            c_dtype=self._traits.c_dtype,
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

    def __repr__(self) -> str:
        return (
            f"TileMmaPlan(shape={self._shape}, atom={self._atom_shape}, "
            f"a={self._a_dtype!r}, b={self._b_dtype!r}, c={self._c_dtype!r}, "
            f"d={self._d_dtype!r}, target={self._target!r}, op_id={self.op_id!r})"
        )
