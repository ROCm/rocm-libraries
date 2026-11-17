# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ROCKE architecture target metadata (the polymorphic-core SSOT).

``ArchTarget`` is the single ROCKE-owned description of *what a gfx target
supports* — wave size, LDS capacity, the MMA atom catalog, memory capability
bits, and resource limits. It carries **hardware facts only**: no pipeline or
scheduler vocabulary (those are instance-side policy), and no LLVM intrinsic
text (that is the ``ISABackend``).

The data lives in ``core/arch/data/arch_specs.json`` and is loaded here. Nothing
in this module imports from ``dispatcher/`` — ROCKE is standalone and must be
importable/testable without the dispatcher tree on disk.

See ``dsl_docs/architecture/multi_arch_data_layout.md``.
"""

from __future__ import annotations

import copy
import json
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Re-export for callers using the original architecture entry point.
from ..dtypes import normalize_dtype
from ..llvm_profile import measured_flavor_for_llvm
from ..storage import MatrixFragmentLayout
from . import domain as arch_domain
from .mfma_scale import (
    eight_bit_matrix_format_pairs,
    matrix_operand_layout,
    narrow_matrix_format_pairs,
)

_DATA_FILE = Path(__file__).parent / "data" / "arch_specs.json"

# A bounded family declaration, rather than a general catalog templating DSL.
# Version 1 freezes the shipped axes and historical enumeration order.
_GFX1250_SCALED_FAMILY = {
    "generate": "gfx1250_wmma_scaled_v1",
    "family": "wmma_scaled",
    "input_dtypes": ["fp8", "bf8", "fp6", "bf6", "fp4"],
    "scale_dtypes": ["e8m0", "e5m3", "e4m3"],
    "block_k": [32, 16],
    "shape": [16, 16, 128],
    "c": "fp32",
}


def _gfx1250_scale_combination_valid(a: str, b: str, sa: str, sb: str) -> bool:
    """Supported matrix/scale combinations, independent of compiler version.

    Mirrors LLVM's isValidWMMAScaleFmtCombination:
    https://github.com/llvm/llvm-project/commit/7597c0c938b364306cbee21dd03835ff5b699c74
    """
    a, b = normalize_dtype(a), normalize_dtype(b)
    return (
        (a == "fp4e2m1" or sa == "e8m0")
        and (b == "fp4e2m1" or sb == "e8m0")
        and (a != "fp4e2m1" or b != "fp4e2m1" or sa == sb)
    )


def _expand_mma_rows(gfx: str, declarations: list[dict]) -> list[dict]:
    """Expand supported families to ordinary rows; reject duplicate identities."""
    if not isinstance(declarations, list):
        raise ValueError(f"MMA declarations for {gfx} must be a list")
    out = []
    ids = set()
    for declaration in declarations:
        if not isinstance(declaration, dict) or (
            "generate" not in declaration and "op_id" not in declaration
        ):
            raise ValueError(f"invalid MMA declaration for {gfx}: {declaration!r}")
        if "generate" not in declaration:
            rows = [declaration]
        else:
            # JSON text comparison also rejects bools masquerading as integers.
            # Admitted by content: any arch may declare the family.
            if json.dumps(declaration, sort_keys=True) != json.dumps(
                _GFX1250_SCALED_FAMILY, sort_keys=True
            ):
                raise ValueError(
                    f"unsupported MMA family declaration for {gfx}: {declaration!r}"
                )
            rows = _gfx1250_scaled_rows(declaration)
        for row in rows:
            if row["op_id"] in ids:
                raise ValueError(f"duplicate MMA op_id for {gfx}: {row['op_id']}")
            ids.add(row["op_id"])
            out.append(row)
    return out


def _gfx1250_scaled_rows(declaration: dict) -> list[dict]:
    inputs = declaration["input_dtypes"]
    scales = declaration["scale_dtypes"]
    blocks = declaration["block_k"]
    # Original homogeneous E8M0 atoms precede mixed E8M0 and supported FP4
    # alternative encodings.
    homogeneous = [
        (0, 32),
        (2, 32),
        (3, 32),
        (1, 32),
        (0, 16),
        (2, 16),
        (3, 16),
        (4, 32),
        (4, 16),
        (1, 16),
    ]
    keys = [(inputs[i], inputs[i], scales[0], scales[0], k) for i, k in homogeneous]
    keys.extend(
        (a, b, scales[0], scales[0], k)
        for k in blocks
        for a in inputs
        for b in inputs
        if a != b
    )
    for k in blocks:
        keys.extend(
            (a, inputs[4], scales[0], sb, k) for a in inputs[:4] for sb in scales[1:]
        )
        keys.extend(
            (inputs[4], b, sa, scales[0], k) for b in inputs[:4] for sa in scales[1:]
        )
        keys.extend((inputs[4], inputs[4], scale, scale, k) for scale in scales[1:])
    rows = []
    # Preserve the supported atoms' historical order and reject unsupported pairs.
    for a, b, sa, sb, block_k in dict.fromkeys(keys):
        if not _gfx1250_scale_combination_valid(a, b, sa, sb):
            continue
        m, n, k = declaration["shape"]
        rows.append(
            {
                "family": declaration["family"],
                "a": a,
                "b": b,
                "c": declaration["c"],
                "a_scale_dtype": sa,
                "b_scale_dtype": sb,
                "scale_block_k": block_k,
                "m": m,
                "n": n,
                "k": k,
                "op_id": f"wmma_gfx1250_f32_{m}x{n}x{k}_{a}_{b}_scale_{sa}_{sb}_k{block_k}",
            }
        )
    return rows


@lru_cache(maxsize=1)
def _load_doc() -> dict:
    # Pin UTF-8 for embedded interpreters whose default codec is ASCII.
    with open(_DATA_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_row(
    arches: dict[str, dict], generic_targets: dict[str, tuple[str, ...]], gfx: str
) -> dict:
    """The raw row whose facts ``gfx`` uses: its own, or the one it points at.

    A row either holds facts or is a pointer, ``{"facts_of": <generic>}`` plus an
    optional ``_comment``. Every member of a supported generic target has a row,
    so rocKE supports it by name; a member whose recorded facts are all its
    generic's is a pointer, so those facts keep one home: the generic target that
    ``generic_targets`` lists it under. A pointer must name exactly that generic,
    and the generic must hold facts itself (one level), so every fact is written
    once and found in one step.
    """
    row = arches[gfx]
    if "facts_of" not in row:
        return row
    source = row["facts_of"]
    own_facts = sorted(key for key in row if key not in ("facts_of", "_comment"))
    if own_facts:
        raise ValueError(
            f"{_DATA_FILE.name} row {gfx!r} has 'facts_of' and its own facts "
            f"{own_facts} -- a row either holds facts or names the row that does"
        )
    if source not in arches:
        raise ValueError(
            f"{_DATA_FILE.name} row {gfx!r} takes its facts from {source!r}, "
            f"which is not a row"
        )
    if "facts_of" in arches[source]:
        raise ValueError(
            f"{_DATA_FILE.name} row {gfx!r} takes its facts from {source!r}, "
            f"which itself takes them from {arches[source]['facts_of']!r} -- "
            f"reference the row that holds the facts"
        )
    listed_under = next(
        (generic for generic, members in generic_targets.items() if gfx in members),
        None,
    )
    if source != listed_under:
        where = repr(listed_under) if listed_under else "no generic target"
        raise ValueError(
            f"{_DATA_FILE.name} row {gfx!r} takes its facts from {source!r}, "
            f"but generic_targets lists {gfx!r} under {where} -- a row may take "
            f"its facts only from the generic target that lists it"
        )
    return arches[source]


@lru_cache(maxsize=1)
def _load_pointers() -> dict[str, str]:
    """Each pointer row's gfx id and the generic target it takes its facts from.

    Every pointer is validated (:func:`_resolve_row`) before it is returned, so
    this is the one place ``facts_of`` is read: no reader sees a malformed
    pointer, whichever loader it reaches first.
    """
    arches = _load_doc()["arches"]
    generic_targets = _load_generic_targets()
    pointers: dict[str, str] = {}
    for gfx, row in arches.items():
        if "facts_of" in row:
            _resolve_row(arches, generic_targets, gfx)
            pointers[gfx] = row["facts_of"]
    return pointers


@lru_cache(maxsize=1)
def _load_specs() -> dict[str, dict]:
    """Every arch row by gfx id, resolved: no returned row is a pointer.

    The whole file is validated on first load. Fact rows have their MMA
    declarations expanded; a pointer row becomes an independent copy of the
    expanded row it points at (so a caller that edits one target never edits
    another), keeping its own ``_comment``. The cached JSON document is not
    modified. Key order follows the file, which the first-hit op_id lookups
    rely on.
    """
    arches = _load_doc()["arches"]
    pointers = _load_pointers()
    expanded = {
        gfx: {**row, "mma": _expand_mma_rows(gfx, row["mma"])}
        for gfx, row in arches.items()
        if gfx not in pointers
    }
    resolved: dict[str, dict] = {}
    for gfx, row in arches.items():
        if gfx in expanded:
            resolved[gfx] = expanded[gfx]
            continue
        facts = copy.deepcopy(expanded[pointers[gfx]])
        facts.pop("_comment", None)
        if "_comment" in row:
            facts["_comment"] = row["_comment"]
        resolved[gfx] = facts
    return resolved


@lru_cache(maxsize=1)
def _load_generic_targets() -> Dict[str, Tuple[str, ...]]:
    return {
        generic: tuple(members)
        for generic, members in _load_doc()["generic_targets"].items()
        if not generic.startswith("_")
    }


# A callable that, given an :class:`~rocke.core.ir.IRBuilder`, a runtime lane
# ``Value`` (0..wave_size-1) and a compile-time fragment slot index, emits the
# index arithmetic for the two tile coordinates of that fragment element and
# returns them as a ``(Value, Value)`` pair. The coordinate *meaning* depends on
# the operand role (see :class:`LayoutMap.role`):
#
#   * accumulator (C/D)  -> ``(row, col)``  in the atom's M x N output tile
#   * A operand          -> ``(row, k)``    in the atom's M x K input tile
#   * B operand          -> ``(k, col)``    in the atom's K x N input tile
#
# The map is *pure* with respect to the IR: it only emits ``arith.*`` ops via the
# builder, so the same map drives both fragment loads (compute the source
# coordinate, gather into the fragment slot) and fragment stores (compute the
# destination coordinate, scatter the slot).
_LaneCoordFn = Any  # Callable[[IRBuilder, Value, int], Tuple[Value, Value]]


@dataclass(frozen=True)
class LayoutMap:
    """Lane/slot -> tile-coordinate map for one MMA fragment role.

    A ``LayoutMap`` is the data-layout contract between an MMA atom and the
    kernel body: it converts a *physical* register location (which lane holds
    the value, and which slot of that lane's fragment vector) into the *logical*
    matrix coordinate that location represents. Because MFMA (wave64) and WMMA
    (wave32) scatter the same logical tile across lanes very differently, the
    kernel cannot hard-code the math — it asks the atom's ``LayoutMap`` for it.

    Attributes
    ----------
    role
        Catalog roles are ``"src0"`` (A, coords ``(row, k)``), ``"src1"``
        (B, coords ``(k, col)``), ``"src2"`` (accumulator input) and ``"dst"``
        (result), whose coords are ``(row, col)``. ``"scale_src0"`` and
        ``"scale_src1"`` use ``(row, K-group)`` and ``(K-group, col)``.
        Scale slots count logical elements,
        independently of the backend register carrier.
    frag_len
        Number of fragment slots per lane for this role (the per-lane vector
        length: e.g. 4 for an MFMA 16x16x16 accumulator, 8 for the WMMA
        accumulator).
    wave_size
        The wave size the map's lane arithmetic assumes (64 for MFMA, 32 for
        WMMA). Surfaced so a kernel can assert the launch geometry matches.
    fn
        The lowering callable ``(builder, lane, slot) -> (coord0, coord1)``.

    Use :meth:`coord` to invoke the map; it validates the slot index and emits
    the arithmetic through the supplied builder.
    """

    role: str
    frag_len: int
    wave_size: int
    fn: _LaneCoordFn = field(repr=False)

    def coord(self, builder: Any, lane: Any, slot: int) -> Tuple[Any, Any]:
        """Emit the index math for ``(lane, slot)`` and return the coord pair.

        ``builder`` is an :class:`~rocke.core.ir.IRBuilder`; ``lane`` is a
        runtime ``i32`` ``Value``; ``slot`` is a compile-time int in
        ``[0, frag_len)``. The returned pair is two ``i32`` ``Value``s whose
        meaning is set by :attr:`role`.
        """
        if not (0 <= slot < self.frag_len):
            raise ValueError(
                f"fragment slot {slot} out of range [0, {self.frag_len}) "
                f"for {self.role!r} layout map"
            )
        return self.fn(builder, lane, slot)


class MmaScaleDType(str, Enum):
    """Scale value formats, independent of matrix dtypes and target support.

    E5M3 is an unsigned scale format, distinct from the signed E5M2 matrix
    format. Only E4M3 shares the accepted matrix spelling ``fp8e4m3``.
    """

    E8M0 = "e8m0"
    E4M3 = "e4m3"
    E5M3 = "e5m3"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> MmaScaleDType | None:
        if value == "fp8e4m3":
            return cls.E4M3
        return None


class MmaScaleBlockK(IntEnum):
    """Number of K elements sharing one scale, common to both matrix inputs."""

    K16 = 16
    K32 = 32


class MmaEmulationRecipe(str, Enum):
    """How an emulated atom is expanded into physical matrix instructions."""

    # Three BF16 MMAs over a Dekker-style split of the fp32 payload each TF32
    # operand carries into a bf16 "big" part and a bf16 "small" remainder,
    # threading one fp32 accumulator. All 32 payload bits are used -- unlike a
    # native XF32 instruction, nothing is dropped from the operand -- so the
    # result tracks the exact product of the payload: more accurate than the
    # native atom on raw fp32 operands. On operands already rounded to TF32
    # precision the native atom is exact and this is not: the omitted
    # small*small cross term is below 2^-16 of a product (and reaches 99% of
    # that at an operand halfway between two bf16 values), the same order as
    # the split's own representation error.
    #
    # Every 32-bit payload splits: the leading bf16 is capped at the largest
    # finite value so the top 2^-9 of the finite range does not round to an
    # infinity, and an infinity or NaN makes every output it feeds non-finite.
    BF16X3 = "bf16x3"

    def __str__(self) -> str:
        return self.value

    @property
    def operand_dtype(self) -> str:
        """The logical dtype of the A/B operands this recipe emulates."""
        return "tf32"


@dataclass(frozen=True)
class MmaEmulation:
    """An atom's expansion into a physical atom, when it has no native encoding.

    ``physical_op_id`` names the atom actually issued. It is the SSOT for the
    emulated atom's fragment lengths and layout maps: those are *derived* from
    it at load time rather than restated, so the logical row and the instruction
    it lowers to can never drift apart. Only the element type differs between
    them -- the logical A fragment occupies the same ``(row, k)`` coordinates.
    """

    recipe: MmaEmulationRecipe
    physical_op_id: str

    def __post_init__(self) -> None:
        # Spelled recipes are validated here rather than at use: an unknown one
        # must fail at catalog load, not silently reach the lowering.
        object.__setattr__(self, "recipe", MmaEmulationRecipe(self.recipe))
        if not self.physical_op_id:
            raise ValueError("MMA emulation requires a physical_op_id")


def _normalize_mma_scales(
    a_dtype: str | None, b_dtype: str | None, block_k: int | None
) -> tuple[MmaScaleDType | None, MmaScaleDType | None, MmaScaleBlockK | None]:
    """Validate the complete scale contract, independently of backend support."""
    if a_dtype is None and b_dtype is None and block_k is None:
        return None, None, None
    try:
        a, b = MmaScaleDType(a_dtype), MmaScaleDType(b_dtype)
    except ValueError:
        raise ValueError("MMA scale dtype must be e8m0, e4m3, or e5m3") from None
    if type(block_k) not in (int, MmaScaleBlockK) or block_k not in (16, 32):
        raise ValueError("MMA scale_block_k must be an integer equal to 16 or 32")
    return a, b, MmaScaleBlockK(block_k)


@dataclass(frozen=True)
class MmaScaleOperand:
    """Scale value format and granularity for one matrix source.

    ``dtype`` is one of ``e8m0``, ``e4m3``, or ``e5m3``; ``fp8e4m3`` is an
    alias for ``e4m3``. It describes the scale values independently of the
    source dtype and the backend's packed register carrier. ``block_size``
    is the number of source elements along K sharing one scale value. Only
    16 and 32 are valid: SCALE and SCALE16 use 32 and 16, respectively.
    """

    dtype: MmaScaleDType | str
    block_size: MmaScaleBlockK | int
    frag_len: int = 0
    layout: LayoutMap | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            dtype = MmaScaleDType(self.dtype)
        except ValueError:
            raise ValueError("MMA scale dtype must be e8m0, e4m3, or e5m3") from None
        if type(self.block_size) not in (
            int,
            MmaScaleBlockK,
        ) or self.block_size not in (16, 32):
            raise ValueError(
                "MMA scale block_size must be an integer equal to 16 or 32"
            )
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "block_size", MmaScaleBlockK(self.block_size))
        if type(self.frag_len) is not int or self.frag_len < 0:
            raise ValueError(f"invalid scale fragment length: {self.frag_len!r}")
        if self.layout is not None and (
            not self.frag_len or self.layout.frag_len != self.frag_len
        ):
            raise ValueError("scale layout does not match its fragment metadata")


@dataclass(frozen=True)
class MmaDst:
    """Metadata for the matrix ``dst`` operand.

    A zero ``frag_len`` leaves the width unspecified; ``IRBuilder.mma`` resolves
    it from the atom's catalog entry. The selected instruction determines the
    supported result dtype.
    """

    dtype: str
    frag_len: int = 0
    layout: Optional[LayoutMap] = field(default=None, repr=False, compare=False)

    def require_layout(self, name: str, op: "MmaOp") -> LayoutMap:
        if self.layout is None:
            raise NotImplementedError(
                f"no verified {name!r} layout map for MMA op_id {op.op_id!r} "
                f"({op.m}x{op.n}x{op.k}); add one to "
                f"_MMA_FRAGMENT_INFO before consuming it"
            )
        return self.layout


@dataclass(frozen=True)
class MmaSrc:
    """Metadata for one matrix ``src``, with an optional scale operand."""

    dtype: str
    frag_len: int = 0
    layout: Optional[LayoutMap] = field(default=None, repr=False, compare=False)
    scale: Optional[MmaScaleOperand] = None

    def require_layout(self, name: str, op: "MmaOp") -> LayoutMap:
        if self.layout is None:
            raise NotImplementedError(
                f"no verified {name!r} layout map for MMA op_id {op.op_id!r} "
                f"({op.m}x{op.n}x{op.k}); add one to "
                f"_MMA_FRAGMENT_INFO before consuming it"
            )
        return self.layout


@dataclass(frozen=True)
class MmaOp:
    """A single supported matrix-multiply-accumulate atom on a target.

    ``op_id`` identifies a concrete operand contract. The backend reads
    metadata, not the ID spelling; LLVM intrinsic text belongs to the ISA layer.

    ``srcs`` contains ``src0``, ``src1``, and ``src2`` in machine order. The
    first two are multiplicands, ``src2`` is the accumulator input, and ``dst``
    is the matrix result.
    Optional scale operands belong to the source they scale; they are not extra
    matrix fragments in ``srcs``.
    The ``*_frag_len`` fields and the layout-map accessors describe the
    *physical* register fragmentation of the atom: how many values of each
    operand a single lane holds, and where (in the logical M/N/K tile) each of
    those values lives. They are the cross-arch data-layout contract that lets
    one kernel body drive both MFMA (wave64) and WMMA (wave32). They are derived
    from ``(op_id, family, shape)`` at load time via :data:`_MMA_FRAGMENT_INFO`;
    op_ids without verified layout maps still carry frag lengths but raise from
    the accessor until a map is added.

    ``emulation`` is ``None`` for an atom the hardware encodes directly. When
    set, the atom has no native encoding and the lowering expands it into
    repeated issues of ``emulation.physical_op_id``; the frag lengths and layout
    maps on this row are inherited from that physical atom. Prefer
    :attr:`is_native` over testing the field.
    """

    family: str  # "mma" | "wmma" | "wmma_scaled"
    srcs: Tuple[MmaSrc, MmaSrc, MmaSrc]
    dst: MmaDst
    m: int
    n: int
    k: int
    op_id: str
    wave_size: int = 64

    # ``None`` means natively encoded. See the class docstring; read through
    # :attr:`is_native` when all you need is "does the hardware do this".
    emulation: Optional[MmaEmulation] = None

    def __post_init__(self) -> None:
        if len(self.srcs) != 3:
            raise ValueError(f"MMA op {self.op_id!r} requires exactly 3 matrix sources")
        for i, src in enumerate(self.srcs):
            if src.scale is not None and src.scale.layout is not None:
                layout = src.scale.layout
                if layout.role != f"scale_src{i}" or layout.wave_size != self.wave_size:
                    raise ValueError("scale layout does not match its source metadata")
        if self.family == "wmma_scaled":
            _normalize_mma_scales(
                self.a_scale_dtype, self.b_scale_dtype, self.scale_block_k
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.m, self.n, self.k)

    def src(self, index: int) -> MmaSrc:
        if not 0 <= index < len(self.srcs):
            raise IndexError(f"MMA src index {index} is outside [0, {len(self.srcs)})")
        return self.srcs[index]

    def src_layout(self, index: int) -> LayoutMap:
        return self.src(index).require_layout(f"src{index}", self)

    def dst_layout(self) -> LayoutMap:
        return self.dst.require_layout("dst", self)

    # A/B/C project the three sources, matching the query arguments. Result
    # consumers use dst explicitly, including when its dtype or maps differ.
    @property
    def a_dtype(self) -> str:
        return self.srcs[0].dtype

    @property
    def b_dtype(self) -> str:
        return self.srcs[1].dtype

    @property
    def c_dtype(self) -> str:
        return self.srcs[2].dtype

    @property
    def dst_dtype(self) -> str:
        """The independent instruction result dtype."""
        return self.dst.dtype

    @property
    def a_frag_len(self) -> int:
        return self.srcs[0].frag_len

    @property
    def b_frag_len(self) -> int:
        return self.srcs[1].frag_len

    @property
    def c_frag_len(self) -> int:
        return self.srcs[2].frag_len

    # --- physical-layout compatibility accessors -------------------------
    @property
    def is_native(self) -> bool:
        """True when the hardware encodes this atom directly (no expansion)."""
        return self.emulation is None

    # --- physical-layout accessors ---------------------------------------
    def a_layout(self) -> LayoutMap:
        """The A-operand ``(row, k)`` lane/slot -> coordinate map."""
        return self.src_layout(0)

    def b_layout(self) -> LayoutMap:
        """The B-operand ``(k, col)`` lane/slot -> coordinate map."""
        return self.src_layout(1)

    def c_layout(self) -> LayoutMap:
        """Accumulator input ``(row, col)`` lane/slot -> coordinate map."""
        return self.src_layout(2)

    @property
    def a_scale_dtype(self) -> MmaScaleDType | None:
        return self.srcs[0].scale.dtype if self.srcs[0].scale else None

    @property
    def b_scale_dtype(self) -> MmaScaleDType | None:
        return self.srcs[1].scale.dtype if self.srcs[1].scale else None

    @property
    def scale_block_k(self) -> MmaScaleBlockK | None:
        """Shared block size for consumers requiring a complete A/B contract."""
        a, b = self.srcs[0].scale, self.srcs[1].scale
        if a is None and b is None:
            return None
        if a is None or b is None or a.block_size != b.block_size:
            raise ValueError("MMA sources do not have a shared scale block size")
        return a.block_size

    def has_ab_layouts(self) -> bool:
        """Whether both operand maps are registered, so a kernel can load A and B.

        A catalog row can name an instruction before its lane maps are verified;
        a validator that admits such an atom hands the builder an op it cannot
        consume.
        """
        return self.srcs[0].layout is not None and self.srcs[1].layout is not None

    @property
    def a_scale_frag_len(self) -> int:
        return self.srcs[0].scale.frag_len if self.srcs[0].scale else 0

    @property
    def b_scale_frag_len(self) -> int:
        return self.srcs[1].scale.frag_len if self.srcs[1].scale else 0

    def src_scale_layout(self, index: int) -> LayoutMap:
        scale = self.src(index).scale
        if scale is None or scale.layout is None:
            raise NotImplementedError(
                f"no verified 'scale_src{index}' layout map for MMA op_id {self.op_id!r} "
                f"({self.m}x{self.n}x{self.k}); add one to "
                f"_MMA_FRAGMENT_INFO before consuming it"
            )
        return scale.layout

    def a_scale_layout(self) -> LayoutMap:
        return self.src_scale_layout(0)

    def b_scale_layout(self) -> LayoutMap:
        return self.src_scale_layout(1)

    def acc_layout(self) -> LayoutMap:
        return self.src_layout(2)


# ---------------------------------------------------------------------------
# Physical fragment layout maps
# ---------------------------------------------------------------------------
#
# The lane arithmetic below is the BYTE-IDENTICAL contract for the MFMA path: it
# reproduces exactly the math the existing MFMA kernels already emit, namely
# ``helpers/atoms.py::MfmaAtom.lane_to_output`` (accumulator) and the operand
# layout documented on the ``MfmaAtom`` factory methods. The WMMA maps encode
# the hardware-verified wave32 gfx1151 layout (see ``instances/gfx1151``).
#
# Each builder returns a ``(builder, lane, slot) -> (coord0, coord1)`` closure.
# ``slot`` is bound at closure-build time? No — it is passed at call time so a
# single LayoutMap covers all slots of the fragment.


def _mfma_row_col_16x16(builder, lane, slot):
    """MFMA 16x16 accumulator: slot ``i`` -> ``(m_blk*4 + i, lane % 16)``.

    Mirrors ``MfmaAtom.lane_to_output`` for the (16, 16) case with
    ``c_per_lane == 4`` (``row = m_blk * 4 + i``, ``col = lane % 16``).
    """
    c16 = builder.const_i32(16)
    n_in_atom = builder.mod(lane, c16)
    m_blk = builder.div(lane, c16)
    row = builder.add(builder.mul(m_blk, builder.const_i32(4)), builder.const_i32(slot))
    return row, n_in_atom


def _mfma_row_col_32x32(builder, lane, slot):
    """MFMA 32x32 accumulator (``c_per_lane == 16``).

    Mirrors ``MfmaAtom.lane_to_output`` for the (32, 32) case:
    ``row = (i // 4) * 8 + (lane // 32) * 4 + (i % 4)``, ``col = lane % 32``.
    """
    c32 = builder.const_i32(32)
    n_in_atom = builder.mod(lane, c32)
    m_blk = builder.div(lane, c32)
    rb = slot // 4
    ri = slot % 4
    row = builder.add(
        builder.add(
            builder.const_i32(rb * 8), builder.mul(m_blk, builder.const_i32(4))
        ),
        builder.const_i32(ri),
    )
    return row, n_in_atom


def _mfma_a_16x16x4(builder, lane, slot):
    """MFMA 16x16x4 fp32 / f64 A operand: each lane holds one scalar.

    row = lane % 16, k = lane // 16 (slot is always 0 — a_frag_len == 1).
    Returns ``(row, k)``.
    """
    c16 = builder.const_i32(16)
    m_in_atom = builder.mod(lane, c16)
    k = builder.div(lane, c16)
    return m_in_atom, k


def _mfma_b_16x16x4(builder, lane, slot):
    """MFMA 16x16x4 fp32 / f64 B operand: each lane holds one scalar.

    col = lane % 16, k = lane // 16 (slot is always 0 — b_frag_len == 1).
    Returns ``(k, col)``.
    """
    c16 = builder.const_i32(16)
    n_in_atom = builder.mod(lane, c16)
    k = builder.div(lane, c16)
    return k, n_in_atom


def _mfma_a_32x32x2_f32(builder, lane, slot):
    """MFMA 32x32x2 fp32 A operand: each lane holds one fp32 scalar.

    row = lane % 32, k = lane // 32 (slot is always 0 — a_frag_len == 1).
    Returns ``(row, k)``.
    """
    c32 = builder.const_i32(32)
    m_in_atom = builder.mod(lane, c32)
    k = builder.div(lane, c32)
    return m_in_atom, k


def _mfma_b_32x32x2_f32(builder, lane, slot):
    """MFMA 32x32x2 fp32 B operand: each lane holds one fp32 scalar.

    col = lane % 32, k = lane // 32 (slot is always 0 — b_frag_len == 1).
    Returns ``(k, col)``.
    """
    c32 = builder.const_i32(32)
    n_in_atom = builder.mod(lane, c32)
    k = builder.div(lane, c32)
    return k, n_in_atom


def _mfma_row_col_16x16x4_f64(builder, lane, slot):
    """MFMA 16x16x4 f64 accumulator: slot ``i`` -> ``(i*4 + lane // 16, lane % 16)``.

    Unlike ``_mfma_row_col_16x16``, consecutive slots of a lane are 4 rows apart.
    """
    c16 = builder.const_i32(16)
    n_in_atom = builder.mod(lane, c16)
    m_blk = builder.div(lane, c16)
    row = builder.add(builder.const_i32(slot * 4), m_blk)
    return row, n_in_atom


def _mfma_b_4x4x4_f64(builder, lane, slot):
    """MFMA 4x4x4 f64 B operand (4 blocks, one scalar per lane).

    Blocks are stacked along rows: lane ``16*k + 4*blk + n`` holds
    ``(blk*4 + k, n)``. Returns ``(blk*4 + k, n)``.
    """
    c4 = builder.const_i32(4)
    n_in_blk = builder.mod(lane, c4)
    blk = builder.mod(builder.div(lane, c4), c4)
    k = builder.div(lane, builder.const_i32(16))
    row = builder.add(builder.mul(blk, c4), k)
    return row, n_in_blk


def _mfma_row_col_4x4x4_f64(builder, lane, slot):
    """MFMA 4x4x4 f64 accumulator (4 blocks, one scalar per lane).

    Blocks are stacked along rows: lane ``16*m + 4*blk + n`` holds
    ``(blk*4 + m, n)``. Returns ``(blk*4 + m, n)``.
    """
    c4 = builder.const_i32(4)
    n_in_blk = builder.mod(lane, c4)
    blk = builder.mod(builder.div(lane, c4), c4)
    m = builder.div(lane, builder.const_i32(16))
    row = builder.add(builder.mul(blk, c4), m)
    return row, n_in_blk


def _mfma_operand_map(rows: int, k_per_lane: int):
    """The A and B lane maps of a dense MFMA whose A has ``rows`` rows.

    Lane ``l`` holds row ``l % rows`` of A (column ``l % rows`` of B), and its
    ``k_per_lane`` slots hold the contiguous K run that starts at
    ``(l // rows) * k_per_lane``::

        16x16x16 i8 (rows 16, k_per_lane 4):
          lanes  0..15 -> K 0..3     lanes 16..31 -> K 4..7
          lanes 32..47 -> K 8..11    lanes 48..63 -> K 12..15

    Each lane's K run is one contiguous block: for the K-packed CDNA3 16x16x32
    f16 atom this packing is hardware-verified to 1e-3, while the flat-concat
    alternative (``[k_blk*4 : k_blk*4+4] + [k_blk*4+16 : k_blk*4+20]``) validates
    only to 1e-2. Every f16, bf16, xf32, fp8, bf8 and i8 MFMA uses this map; the
    f32 and f64 atoms, one element per lane, use :func:`_mfma_a_16x16x4` and
    :func:`_mfma_a_32x32x2_f32` and their B twins.

    Returns ``(a_map, b_map)``: ``a_map`` gives ``(row, k)``, ``b_map`` gives
    ``(k, col)``. The C++ twin is ``_mfma_operand_a`` / ``_mfma_operand_b`` in
    ``cpp/core/arch/data.cpp``.
    """

    def a_map(builder, lane, slot):
        c_rows = builder.const_i32(rows)
        row = builder.mod(lane, c_rows)
        k_blk = builder.div(lane, c_rows)
        k = builder.add(
            builder.mul(k_blk, builder.const_i32(k_per_lane)), builder.const_i32(slot)
        )
        return row, k

    def b_map(builder, lane, slot):
        col, k = a_map(builder, lane, slot)
        return k, col

    return a_map, b_map


def _wmma_row_col_16x16(builder, lane, slot):
    """WMMA 16x16x16 accumulator (wave32, hardware-verified gfx1151).

    ``<8 x float>`` per lane; slot ``i`` -> ``(row 2*i + lane // 16,
    col lane % 16)``. Returns ``(row, col)``.
    """
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    half = builder.div(lane, c16)
    row = builder.add(builder.const_i32(2 * slot), half)
    return row, col


def _wmma_a_16x16(builder, lane, slot):
    """WMMA 16x16x16 A operand (wave32): lane ``l`` holds row ``l % 16``;
    the ``<16 x half>`` fragment slot ``i`` is K=``i`` (0..15). Returns
    ``(row, k)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    return row, builder.const_i32(slot)


def _wmma_b_16x16(builder, lane, slot):
    """WMMA 16x16x16 B operand (wave32): lane ``l`` holds col ``l % 16``;
    fragment slot ``i`` is K=``i`` (0..15). Returns ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    return builder.const_i32(slot), col


# --- RDNA3/3.5 integer WMMA (iu8) 16x16x16 lane maps --------------------------
# Same lane geometry as f16 WMMA (lane l holds row/col l%16, cross-half
# duplication), but the K=16 dimension is *packed* into the <4 x i32> A/B
# fragment: slot j (0..3) is one i32 holding the four int8 K-values
# [4j, 4j+1, 4j+2, 4j+3]. The lane map therefore returns the K *base* of the
# slot (4*j); the kernel/staging code packs the four consecutive K bytes from
# there. The accumulator is identical to f16 WMMA (_wmma_row_col_16x16), only the
# element type is i32 instead of f32.
def _wmma_a_16x16_iu8(builder, lane, slot):
    """iu8 WMMA A operand (wave32): lane ``l`` holds row ``l % 16``; A fragment
    slot ``j`` is the i32 packing K=[4j..4j+3]. Returns ``(row, k_base=4j)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    return row, builder.const_i32(4 * slot)


def _wmma_b_16x16_iu8(builder, lane, slot):
    """iu8 WMMA B operand (wave32): lane ``l`` holds col ``l % 16``; B fragment
    slot ``j`` is the i32 packing K=[4j..4j+3]. Returns ``(k_base=4j, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    return builder.const_i32(4 * slot), col


# --- RDNA3/3.5 integer WMMA (iu4) 16x16x16 lane maps --------------------------
# Same lane geometry as iu8, but the K=16 dimension is packed into <2 x i32>:
# slot j (0..1) holds eight signed int4 values K=[8j..8j+7].
def _wmma_a_16x16_iu4(builder, lane, slot):
    """iu4 WMMA A operand (wave32): lane ``l`` holds row ``l % 16``; A fragment
    slot ``j`` is the i32 packing K=[8j..8j+7]. Returns ``(row, k_base=8j)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    return row, builder.const_i32(8 * slot)


def _wmma_b_16x16_iu4(builder, lane, slot):
    """iu4 WMMA B operand (wave32): lane ``l`` holds col ``l % 16``; B fragment
    slot ``j`` is the i32 packing K=[8j..8j+7]. Returns ``(k_base=8j, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    return builder.const_i32(8 * slot), col


# --- RDNA4 (gfx12) WMMA 16x16x16 lane maps ------------------------------------
# RDNA4 dropped the RDNA3/3.5 cross-half duplication: A/B fragments are
# ``<8 x half>`` per lane (not <16 x half>), and the K dimension is split across
# the two lane-halves (lanes 0-15 carry K 0..7, lanes 16-31 carry K 8..15). The
# accumulator is column-distributed (CDNA/MFMA-style): lanes index columns,
# registers index rows. The WMMA attention numeric suite exercises these maps on
# gfx1200/gfx1201 hardware; examples/gfx1201/wmma_probe.py probes single atoms.
def _wmma_gfx12_row_col_16x16(builder, lane, slot):
    """RDNA4 WMMA 16x16x16 accumulator (wave32): ``<8 x float>`` per lane;
    slot ``i`` -> ``(row (lane // 16) * 8 + i, col lane % 16)``. Returns
    ``(row, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    half = builder.div(lane, c16)
    row = builder.add(builder.mul(half, builder.const_i32(8)), builder.const_i32(slot))
    return row, col


def _wmma_gfx12_a_16x16(builder, lane, slot):
    """RDNA4 WMMA 16x16x16 A operand (wave32): lane ``l`` holds row ``l % 16``;
    the ``<8 x half>`` fragment slot ``i`` is K=``(l // 16) * 8 + i``. Returns
    ``(row, k)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(8)), builder.const_i32(slot))
    return row, k


def _wmma_gfx12_b_16x16(builder, lane, slot):
    """RDNA4 WMMA 16x16x16 B operand (wave32): lane ``l`` holds col ``l % 16``;
    the ``<8 x half>`` fragment slot ``i`` is K=``(l // 16) * 8 + i``. Returns
    ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(8)), builder.const_i32(slot))
    return k, col


# --- gfx1250 (gfx1250) WMMA 16x16x32 lane maps (CDNA, GFX12 programming model) --
# gfx1250 is wave32/WMMA like gfx12 RDNA but the primary fp16/bf16 atom is
# 16x16x32 (K=32, not 16). A/B fragments are <16 x half> per lane (16 K-elements
# each); the K dimension is split across the two lane-halves (lanes 0-15 carry
# K 0..15, lanes 16-31 carry K 16..31). The accumulator is the same 16x16
# column-distributed layout as gfx12 (<8 x float>, slot i -> row (l//16)*8 + i,
# col l%16), since the output tile is still 16x16. These maps are the
# *hypothesis* verified empirically by examples/gfx1250/wmma_probe.py.
# The gfx1250 WMMA 16x16x4 fp32 A and B maps are CK-sourced (wmma_gfx12.hpp
# parameters); not validated on hardware. Lane l holds row l % 16 of A (column
# l % 16 of B) and the two K values 2 * (l // 16) + slot, so lanes 0-15 carry K
# 0..1 and lanes 16-31 carry K 2..3.
def _wmma_gfx1250_a_16x16x4_f32(builder, lane, slot):
    """gfx1250 WMMA 16x16x4 fp32 A operand (wave32): row = lane % 16;
    K = 2 * (lane // 16) + slot (slot in {0, 1}). Returns ``(row, k)``."""
    c2, c16 = builder.const_i32(2), builder.const_i32(16)
    row = builder.mod(lane, c16)
    k = builder.add(builder.mul(builder.div(lane, c16), c2), builder.const_i32(slot))
    return row, k


def _wmma_gfx1250_b_16x16x4_f32(builder, lane, slot):
    """gfx1250 WMMA 16x16x4 fp32 B operand (wave32): col = lane % 16;
    K = 2 * (lane // 16) + slot (slot in {0, 1}). Returns ``(k, col)``."""
    c2, c16 = builder.const_i32(2), builder.const_i32(16)
    col = builder.mod(lane, c16)
    k = builder.add(builder.mul(builder.div(lane, c16), c2), builder.const_i32(slot))
    return k, col


def _wmma_gfx1250_a_16x16x32(builder, lane, slot):
    """gfx1250 WMMA 16x16x32 A operand (wave32): lane ``l`` holds row ``l % 16``;
    the ``<16 x half>`` fragment slot ``i`` is K=``(l // 16) * 16 + i``. Returns
    ``(row, k)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(16)), builder.const_i32(slot))
    return row, k


def _wmma_gfx1250_b_16x16x32(builder, lane, slot):
    """gfx1250 WMMA 16x16x32 B operand (wave32): lane ``l`` holds col ``l % 16``;
    the ``<16 x half>`` fragment slot ``i`` is K=``(l // 16) * 16 + i``. Returns
    ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(builder.mul(k_half, builder.const_i32(16)), builder.const_i32(slot))
    return k, col


def _wmma_gfx1250_a_16x16x64_f8(builder, lane, slot):
    """gfx1250 WMMA 16x16x64 fp8/bf8 A operand (wave32): lane ``l`` holds row
    ``l % 16``; the ``<8 x i32>`` fragment slot ``i`` is the dword covering
    K=``(l // 16) * 32 + i * 4`` (four low-bit bytes per slot, K-contiguous).
    Same lane math as the K=32 f16/bf16 map with the per-lane K span doubled
    from 16 elements to 32. Returns ``(row, k)``."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(
        builder.mul(k_half, builder.const_i32(32)), builder.const_i32(slot * 4)
    )
    return row, k


def _wmma_gfx1250_b_16x16x64_f8(builder, lane, slot):
    """gfx1250 WMMA 16x16x64 fp8/bf8 B operand (wave32): lane ``l`` holds col
    ``l % 16``; the ``<8 x i32>`` fragment slot ``i`` is the dword covering
    K=``(l // 16) * 32 + i * 4`` (four low-bit bytes per slot, K-contiguous).
    Returns ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(
        builder.mul(k_half, builder.const_i32(32)), builder.const_i32(slot * 4)
    )
    return k, col


def _wmma_gfx1250_a_16x16x128_f8(builder, lane, slot):
    """gfx1250 WMMA 16x16x128 8-bit A operand (wave32): lane ``l`` holds row
    ``l % 16``; the ``<16 x i32>`` fragment slot ``i`` is the dword covering
    K=``(l // 16) * 64 + i * 4`` (four bytes per slot, K-contiguous). Same lane
    math as the K=64 fp8 map with the per-lane K span doubled from 32 bytes to
    64. Returns ``(row, k)``.

    The K order is the GEMM's, not necessarily the instruction's: A and B share
    it, and a dot product is the same under any K permutation applied to both,
    so a GEMM only needs the two maps to agree. Verified that way on gfx1250
    silicon (random asymmetric integer A/B reproduce ``C = A @ B.T`` exactly in
    all 256 accumulator cells, which a row/col swap would fail). Do not reuse
    it where the K order itself matters (mixed formats, block scales)."""
    c16 = builder.const_i32(16)
    row = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(
        builder.mul(k_half, builder.const_i32(64)), builder.const_i32(slot * 4)
    )
    return row, k


def _wmma_gfx1250_b_16x16x128_f8(builder, lane, slot):
    """gfx1250 WMMA 16x16x128 8-bit B operand (wave32): lane ``l`` holds col
    ``l % 16``; the ``<16 x i32>`` fragment slot ``i`` is the dword covering
    K=``(l // 16) * 64 + i * 4`` (four low-bit bytes per slot, K-contiguous).
    Returns ``(k, col)``."""
    c16 = builder.const_i32(16)
    col = builder.mod(lane, c16)
    k_half = builder.div(lane, c16)
    k = builder.add(
        builder.mul(k_half, builder.const_i32(64)), builder.const_i32(slot * 4)
    )
    return k, col


def _wmma_gfx1250_a_scale(builder, lane, slot):
    row = builder.mod(lane, builder.const_i32(16))
    return row, builder.const_i32(slot)


def _wmma_gfx1250_b_scale(builder, lane, slot):
    col = builder.mod(lane, builder.const_i32(16))
    return builder.const_i32(slot), col


def integer_mma_op_ids(op_id: str, bits: int) -> Tuple[str, str, str, str]:
    """The op_ids of one integer matrix instruction, one per operand signedness.

    The instruction takes the signedness of A and B as flags, so each
    combination is its own catalog row: ``op_id`` for signed A and B, then
    ``op_id`` with ``_u<bits>_i<bits>`` (A unsigned), ``_i<bits>_u<bits>`` (B
    unsigned) and ``_u<bits>_u<bits>`` appended, for example
    ``wmma_gfx12_i32_16x16x16_iu8_u8_i8``.
    """
    return (
        op_id,
        f"{op_id}_u{bits}_i{bits}",
        f"{op_id}_i{bits}_u{bits}",
        f"{op_id}_u{bits}_u{bits}",
    )


def _f8f6f4_op_ids(op_id: str, pairs) -> dict[str, tuple[str, str]]:
    """``{op_id + "_<a>_<b>": (a, b)}`` for each (A, B) format pair of ``pairs``."""
    return {f"{op_id}_{a}_{b}": (a, b) for a, b in pairs}


def narrow_f8f6f4_op_ids(op_id: str) -> dict[str, tuple[str, str]]:
    """The op_ids of one f8f6f4 instruction, one per format pair it owns.

    Each (A, B) pair of
    :func:`~rocke.core.arch.mfma_scale.narrow_matrix_format_pairs` (a 6-bit or
    4-bit operand) is its own catalog row: ``op_id`` with ``_<a>_<b>`` appended
    in canonical dtype names, for example
    ``wmma_gfx1250_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1``. Returns
    ``{op_id: (a, b)}`` in pair order.
    """
    return _f8f6f4_op_ids(op_id, narrow_matrix_format_pairs())


# The op_id the gfx950 fp8 x fp8 f8f6f4 MFMA at 16x16x128 had before the
# catalog listed the instruction's other formats; a hand-written handler lowers
# it.
GFX950_FP8_16X16X128_OP_ID = "mfma_f32_16x16x128_fp8"


def gfx950_eight_bit_f8f6f4_op_ids(
    shape: tuple[int, int, int],
) -> dict[str, tuple[str, str]]:
    """The op_ids of the gfx950 f8f6f4 MFMA at ``shape`` ``(16, 16, 128)`` or
    ``(32, 32, 64)`` for each pair of
    :func:`~rocke.core.arch.mfma_scale.eight_bit_matrix_format_pairs`.

    Each is ``mfma_f32_<M>x<N>x<K>_f8f6f4`` with ``_<a>_<b>`` appended, for
    example ``mfma_f32_32x32x64_f8f6f4_fp8e4m3_bf8e5m2``, except fp8 x fp8 at
    16x16x128, which is :data:`GFX950_FP8_16X16X128_OP_ID`. Returns
    ``{op_id: (a, b)}`` in pair order.
    """
    m, n, k = shape
    return {
        (
            GFX950_FP8_16X16X128_OP_ID
            if (shape, a, b) == ((16, 16, 128), "fp8e4m3", "fp8e4m3")
            else op_id
        ): (a, b)
        for op_id, (a, b) in _f8f6f4_op_ids(
            f"mfma_f32_{m}x{n}x{k}_f8f6f4", eight_bit_matrix_format_pairs()
        ).items()
    }


def _lane_map_from_fragment_layout(layout: MatrixFragmentLayout, role: str):
    """The ``(builder, lane, slot) -> coords`` lane map of ``layout`` for operand
    ``role`` (``"a"`` or ``"b"``).

    Lane ``l`` slot ``s`` sits at index ``l % lanes_per_group`` (the row of A,
    the column of B) and K ``(l // lanes_per_group) * chunk_elements + c``,
    where ``c = (s // chunk_elements) * lane_groups * chunk_elements +
    s % chunk_elements`` is a constant per slot -- the same coordinates
    :meth:`MatrixFragmentLayout.coord` gives. A returns ``(row, k)``, B
    ``(k, col)``. The C++ twin is ``_fragment_layout_a`` / ``_fragment_layout_b``
    in ``cpp/core/arch/data.cpp``.
    """
    if role not in ("a", "b"):
        raise ValueError(f"a fragment layout maps operand 'a' or 'b', got {role!r}")
    chunk_elements = layout.chunk_elements
    run_stride = layout.lane_groups * chunk_elements

    def placement(builder, lane, slot):
        chunk, element = divmod(slot, chunk_elements)
        c_lanes = builder.const_i32(layout.lanes_per_group)
        index = builder.mod(lane, c_lanes)
        k = builder.add(
            builder.mul(builder.div(lane, c_lanes), builder.const_i32(chunk_elements)),
            builder.const_i32(chunk * run_stride + element),
        )
        return (index, k) if role == "a" else (k, index)

    return placement


def _f8f6f4_operand_maps(a: str, b: str, words: int, lane_groups: int, lanes: int):
    """The A and B lane maps of an f8f6f4 instruction reading ``a`` x ``b``."""
    return (
        _lane_map_from_fragment_layout(
            matrix_operand_layout(a, words, lane_groups, lanes), "a"
        ),
        _lane_map_from_fragment_layout(
            matrix_operand_layout(b, words, lane_groups, lanes), "b"
        ),
    )


@dataclass(frozen=True)
class _FragOperand:
    """Physical fragment metadata for one machine operand position."""

    frag_len: int
    fn: _LaneCoordFn = None


@dataclass(frozen=True)
class _FragInfo:
    """Per-op_id physical metadata for ``srcs[0..2]`` and ``dst``."""

    srcs: Tuple[_FragOperand, _FragOperand, _FragOperand]
    dst: _FragOperand
    wave_size: int
    a_scale_frag_len: int = 0
    b_scale_frag_len: int = 0
    a_scale_fn: _LaneCoordFn = None
    b_scale_fn: _LaneCoordFn = None

    def __post_init__(self) -> None:
        if len(self.srcs) != 3:
            raise ValueError("MMA fragment metadata requires exactly 3 sources")


def _shared_src2_dst_frag_info(
    src0_frag_len: int,
    src1_frag_len: int,
    row_col_frag_len: int,
    wave_size: int,
    src0_fn: _LaneCoordFn = None,
    src1_fn: _LaneCoordFn = None,
    row_col_fn: _LaneCoordFn = None,
    a_scale_frag_len: int = 0,
    b_scale_frag_len: int = 0,
    a_scale_fn: _LaneCoordFn = None,
    b_scale_fn: _LaneCoordFn = None,
) -> _FragInfo:
    """Build metadata whose ``src2`` and ``dst`` layouts currently match.

    ``src2`` and ``dst`` get separate descriptors even when the current ISA
    gives them the same fragment width and row/column coordinate function.
    This is a catalog-construction convenience, not a constraint on
    :class:`_FragInfo`; distinct contracts use that type directly.
    """

    return _FragInfo(
        srcs=(
            _FragOperand(src0_frag_len, src0_fn),
            _FragOperand(src1_frag_len, src1_fn),
            _FragOperand(row_col_frag_len, row_col_fn),
        ),
        dst=_FragOperand(row_col_frag_len, row_col_fn),
        wave_size=wave_size,
        a_scale_frag_len=a_scale_frag_len,
        b_scale_frag_len=b_scale_frag_len,
        a_scale_fn=a_scale_fn,
        b_scale_fn=b_scale_fn,
    )


# op_id -> physical fragment metadata. Frag lengths are populated for every
# IRBuilder MFMA/WMMA atom (so ``IRBuilder.mma`` can size the result vector);
# layout-map functions are populated for the atoms whose lane math is verified.
# Adding a new atom is one row here.
_MMA_FRAGMENT_INFO: Dict[str, _FragInfo] = {
    "mfma_f32_32x32x4_xf32": _shared_src2_dst_frag_info(
        2, 2, 16, 64, *_mfma_operand_map(32, 2), _mfma_row_col_32x32
    ),
    "mfma_f32_16x16x8_xf32": _shared_src2_dst_frag_info(
        2, 2, 4, 64, *_mfma_operand_map(16, 2), _mfma_row_col_16x16
    ),
    # --- MFMA fp32 (wave64) -----------------------------------------------
    # A/B are scalar float per lane (a_frag_len=b_frag_len=1); accumulator
    # shares the standard 16x16 / 32x32 layout (c_frag_len=4 / 16).
    "mfma_f32_16x16x4_f32": _shared_src2_dst_frag_info(
        1, 1, 4, 64, _mfma_a_16x16x4, _mfma_b_16x16x4, _mfma_row_col_16x16
    ),
    "mfma_f32_32x32x2_f32": _shared_src2_dst_frag_info(
        1, 1, 16, 64, _mfma_a_32x32x2_f32, _mfma_b_32x32x2_f32, _mfma_row_col_32x32
    ),
    # --- MFMA fp64 (wave64) -----------------------------------------------
    # A/B are scalar double per lane in the same layout as fp32 16x16x4; the
    # accumulator has its own f64 layout (c_frag_len=4).
    "mfma_f64_16x16x4_f64": _shared_src2_dst_frag_info(
        1, 1, 4, 64, _mfma_a_16x16x4, _mfma_b_16x16x4, _mfma_row_col_16x16x4_f64
    ),
    # 4x4x4 is four independent 4x4x4 blocks stacked along rows (16x4 A, B and
    # C/D). A matches the 16x16x4 A map.
    "mfma_f64_4x4x4_f64": _shared_src2_dst_frag_info(
        1, 1, 1, 64, _mfma_a_16x16x4, _mfma_b_4x4x4_f64, _mfma_row_col_4x4x4_f64
    ),
    # --- MFMA f16 (wave64) ------------------------------------------------
    "mfma_f32_16x16x16_f16": _shared_src2_dst_frag_info(
        4, 4, 4, 64, *_mfma_operand_map(16, 4), _mfma_row_col_16x16
    ),
    "mfma_f32_16x16x32_f16": _shared_src2_dst_frag_info(
        8, 8, 4, 64, *_mfma_operand_map(16, 8), _mfma_row_col_16x16
    ),
    "mfma_f32_32x32x8_f16": _shared_src2_dst_frag_info(
        4, 4, 16, 64, *_mfma_operand_map(32, 4), _mfma_row_col_32x32
    ),
    "mfma_f32_32x32x16_f16": _shared_src2_dst_frag_info(
        8, 8, 16, 64, *_mfma_operand_map(32, 8), _mfma_row_col_32x32
    ),
    "mfma_f32_4x4x4_f16": _shared_src2_dst_frag_info(4, 4, 4, 64),
    "mfma_f32_4x4x4_bf16": _shared_src2_dst_frag_info(4, 4, 4, 64),
    # --- MFMA bf16 (wave64) ----------------------------------------------
    "mfma_f32_16x16x16_bf16": _shared_src2_dst_frag_info(
        4, 4, 4, 64, *_mfma_operand_map(16, 4), _mfma_row_col_16x16
    ),
    "mfma_f32_16x16x32_bf16": _shared_src2_dst_frag_info(
        8, 8, 4, 64, *_mfma_operand_map(16, 8), _mfma_row_col_16x16
    ),
    "mfma_f32_32x32x8_bf16": _shared_src2_dst_frag_info(
        4, 4, 16, 64, *_mfma_operand_map(32, 4), _mfma_row_col_32x32
    ),
    "mfma_f32_32x32x16_bf16": _shared_src2_dst_frag_info(
        8, 8, 16, 64, *_mfma_operand_map(32, 8), _mfma_row_col_32x32
    ),
    # --- MFMA fp8 / bf8 (wave64) -----------------------------------------
    # fp8/bf8 share the f16 operand lane layout (a_per_lane=8, same K-packing);
    # only the element type / intrinsic mangling differ.
    "mfma_f32_16x16x32_fp8": _shared_src2_dst_frag_info(
        8, 8, 4, 64, *_mfma_operand_map(16, 8), _mfma_row_col_16x16
    ),
    "mfma_f32_16x16x32_bf8": _shared_src2_dst_frag_info(
        8, 8, 4, 64, *_mfma_operand_map(16, 8), _mfma_row_col_16x16
    ),
    "mfma_f32_32x32x16_fp8": _shared_src2_dst_frag_info(
        8, 8, 16, 64, *_mfma_operand_map(32, 8), _mfma_row_col_32x32
    ),
    "mfma_f32_32x32x16_bf8": _shared_src2_dst_frag_info(
        8, 8, 16, 64, *_mfma_operand_map(32, 8), _mfma_row_col_32x32
    ),
    # --- MFMA mixed fp8 x bf8 (wave64) -----------------------------------
    "mfma_f32_16x16x32_fp8_bf8": _shared_src2_dst_frag_info(
        8, 8, 4, 64, *_mfma_operand_map(16, 8), _mfma_row_col_16x16
    ),
    "mfma_f32_16x16x32_bf8_fp8": _shared_src2_dst_frag_info(
        8, 8, 4, 64, *_mfma_operand_map(16, 8), _mfma_row_col_16x16
    ),
    "mfma_f32_32x32x16_fp8_bf8": _shared_src2_dst_frag_info(
        8, 8, 16, 64, *_mfma_operand_map(32, 8), _mfma_row_col_32x32
    ),
    "mfma_f32_32x32x16_bf8_fp8": _shared_src2_dst_frag_info(
        8, 8, 16, 64, *_mfma_operand_map(32, 8), _mfma_row_col_32x32
    ),
    # --- MFMA bf16 without the _1k suffix (wave64; two bf16 per lane) -----
    "mfma_f32_16x16x8_bf16": _shared_src2_dst_frag_info(
        2, 2, 4, 64, *_mfma_operand_map(16, 2), _mfma_row_col_16x16
    ),
    "mfma_f32_32x32x4_bf16": _shared_src2_dst_frag_info(
        2, 2, 16, 64, *_mfma_operand_map(32, 2), _mfma_row_col_32x32
    ),
    # --- MFMA i8 -> i32 (wave64) -------------------------------------------
    "mfma_i32_16x16x16_i8": _shared_src2_dst_frag_info(
        4, 4, 4, 64, *_mfma_operand_map(16, 4), _mfma_row_col_16x16
    ),
    "mfma_i32_32x32x8_i8": _shared_src2_dst_frag_info(
        4, 4, 16, 64, *_mfma_operand_map(32, 4), _mfma_row_col_32x32
    ),
    "mfma_i32_16x16x32_i8": _shared_src2_dst_frag_info(
        8, 8, 4, 64, *_mfma_operand_map(16, 8), _mfma_row_col_16x16
    ),
    "mfma_i32_32x32x16_i8": _shared_src2_dst_frag_info(
        8, 8, 16, 64, *_mfma_operand_map(32, 8), _mfma_row_col_32x32
    ),
    "mfma_i32_16x16x64_i8": _shared_src2_dst_frag_info(
        16, 16, 4, 64, *_mfma_operand_map(16, 16), _mfma_row_col_16x16
    ),
    "mfma_i32_32x32x32_i8": _shared_src2_dst_frag_info(
        16, 16, 16, 64, *_mfma_operand_map(32, 16), _mfma_row_col_32x32
    ),
    # --- K=128 f8f6f4 builder ops (wave64), frag lengths only ---------------
    # The unscaled all-format op and the MX op have no catalog row: A/B are 32
    # bytes per lane (<8 x i32> at the intrinsic boundary), accumulator
    # <4 x float>. Registered here so the op_id resolves to correct fragment
    # lengths (this table is the SSOT that ir.IRBuilder.mma consults).
    "mfma_f32_16x16x128_fp4fp6fp8": _shared_src2_dst_frag_info(32, 32, 4, 64),
    "mfma_scale_f32_16x16x128_f8f6f4": _shared_src2_dst_frag_info(32, 32, 4, 64),
    # --- WMMA f16 / bf16 (wave32, RDNA) ----------------------------------
    "wmma_f32_16x16x16_f16": _shared_src2_dst_frag_info(
        16, 16, 8, 32, _wmma_a_16x16, _wmma_b_16x16, _wmma_row_col_16x16
    ),
    # bf16 shares the f16 fragment layout (same 16x16x16 lane math; only the
    # element type / intrinsic mangling differ — operands lower as <16 x i16>).
    "wmma_f32_16x16x16_bf16": _shared_src2_dst_frag_info(
        16, 16, 8, 32, _wmma_a_16x16, _wmma_b_16x16, _wmma_row_col_16x16
    ),
    # --- WMMA f16 -> f16, bf16 -> bf16 (wave32, RDNA3/3.5) ----------------------
    # A and B as for f16 -> f32. The instruction keeps its eight results per
    # lane in the even halves of a <16 x half> (<16 x i16>) register; the
    # catalog row counts the eight logical results, which sit where the f32
    # accumulator's eight slots do (public matrix instruction calculator, op_sel
    # 0), so the result takes the f32 accumulator map.
    "wmma_f16_16x16x16_f16": _shared_src2_dst_frag_info(
        16, 16, 8, 32, _wmma_a_16x16, _wmma_b_16x16, _wmma_row_col_16x16
    ),
    "wmma_bf16_16x16x16_bf16": _shared_src2_dst_frag_info(
        16, 16, 8, 32, _wmma_a_16x16, _wmma_b_16x16, _wmma_row_col_16x16
    ),
    # --- WMMA iu8 (wave32, RDNA3/3.5) ------------------------------------------
    # A/B fragments are <4 x i32> (16 int8 packed 4-per-i32); accumulator is
    # <8 x i32> with the same lane math as the f16 WMMA accumulator.
    # Each signedness combination (integer_mma_op_ids) has the same fragments.
    **{
        op_id: _shared_src2_dst_frag_info(
            4, 4, 8, 32, _wmma_a_16x16_iu8, _wmma_b_16x16_iu8, _wmma_row_col_16x16
        )
        for op_id in integer_mma_op_ids("wmma_i32_16x16x16_iu8", 8)
    },
    # --- WMMA iu4 (wave32, RDNA3/3.5) ------------------------------------------
    # A/B fragments are <2 x i32> (16 int4 packed 8-per-i32); accumulator is
    # <8 x i32> with the same lane math as the f16 WMMA accumulator.
    **{
        op_id: _shared_src2_dst_frag_info(
            2, 2, 8, 32, _wmma_a_16x16_iu4, _wmma_b_16x16_iu4, _wmma_row_col_16x16
        )
        for op_id in integer_mma_op_ids("wmma_i32_16x16x16_iu4", 4)
    },
    # --- WMMA f16 / bf16 (wave32, RDNA4 / gfx12) -------------------------------
    # No cross-half duplication: A/B are <8 x half> per lane; column-distributed
    # accumulator. Lane maps verified by examples/gfx1201/wmma_probe.py.
    "wmma_gfx12_f32_16x16x16_f16": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    "wmma_gfx12_f32_16x16x16_bf16": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    # --- WMMA fp8 / bf8 -> f32, i8 -> i32, f16 -> f16, bf16 -> bf16 (gfx12) --
    # Eight values per lane on the f16 atom's maps: A row lane % 16 with the K
    # run (lane // 16) * 8 + slot, B the same by column, and the accumulator
    # (or 16-bit result) slot i at row (lane // 16) * 8 + i. The public matrix
    # instruction calculator gives these A and B maps exactly for the 8-bit
    # ops; for the two 16-bit-output ops, as for the f16 atom itself, it orders
    # K within a lane differently but identically for A and B, which leaves the
    # product unchanged. The result map is exact for all seven.
    "wmma_gfx12_f32_16x16x16_fp8_fp8": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    "wmma_gfx12_f32_16x16x16_fp8_bf8": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    "wmma_gfx12_f32_16x16x16_bf8_fp8": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    "wmma_gfx12_f32_16x16x16_bf8_bf8": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    **{
        op_id: _shared_src2_dst_frag_info(
            8,
            8,
            8,
            32,
            _wmma_gfx12_a_16x16,
            _wmma_gfx12_b_16x16,
            _wmma_gfx12_row_col_16x16,
        )
        for op_id in integer_mma_op_ids("wmma_gfx12_i32_16x16x16_iu8", 8)
    },
    # --- WMMA i4 -> i32 (wave32, gfx12), every signedness combination --------
    # Eight (K 16) or sixteen (K 32) i4 per lane, packed in one or two words:
    # lane l holds row l % 16 of A (column l % 16 of B) and the contiguous K
    # run starting at (l // 16) * 8 or (l // 16) * 16, slots counted in i4
    # values from the low bits of the first word. The public matrix instruction
    # calculator gives these maps exactly. The K-32 map is the one the gfx1250
    # 16x16x32 atoms use; the result map is the gfx12 accumulator's.
    **{
        op_id: _shared_src2_dst_frag_info(
            8,
            8,
            8,
            32,
            _wmma_gfx12_a_16x16,
            _wmma_gfx12_b_16x16,
            _wmma_gfx12_row_col_16x16,
        )
        for op_id in integer_mma_op_ids("wmma_gfx12_i32_16x16x16_iu4", 4)
    },
    **{
        op_id: _shared_src2_dst_frag_info(
            16,
            16,
            8,
            32,
            _wmma_gfx1250_a_16x16x32,
            _wmma_gfx1250_b_16x16x32,
            _wmma_gfx12_row_col_16x16,
        )
        for op_id in integer_mma_op_ids("wmma_gfx12_i32_16x16x32_iu4", 4)
    },
    "wmma_gfx12_f16_16x16x16_f16": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    "wmma_gfx12_bf16_16x16x16_bf16": _shared_src2_dst_frag_info(
        8, 8, 8, 32, _wmma_gfx12_a_16x16, _wmma_gfx12_b_16x16, _wmma_gfx12_row_col_16x16
    ),
    # --- WMMA fp32 (wave32, gfx1250, CDNA) --------------------------------
    # K=4 atom: A/B are <2 x fp32> per lane (a_frag_len=b_frag_len=2): row (or
    # column) lane % 16, K = 2 * (lane // 16) + slot. CK-sourced
    # (wmma_gfx12.hpp parameters); not validated on hardware. Accumulator is
    # the gfx12 column-distributed <8 x float> (c_frag_len=8).
    "wmma_gfx1250_f32_16x16x4_f32": _shared_src2_dst_frag_info(
        2,
        2,
        8,
        32,
        _wmma_gfx1250_a_16x16x4_f32,
        _wmma_gfx1250_b_16x16x4_f32,
        _wmma_gfx12_row_col_16x16,
    ),
    # --- WMMA f16 / bf16 (wave32, gfx1250, CDNA) ----------------------
    # K=32 atom: A/B are <16 x half> per lane (K split across lane-halves, 16
    # each); accumulator is the same 16x16 column-distributed <8 x float> as
    # gfx12. Lane maps verified by examples/gfx1250/wmma_probe.py.
    "wmma_gfx1250_f32_16x16x32_f16": _shared_src2_dst_frag_info(
        16,
        16,
        8,
        32,
        _wmma_gfx1250_a_16x16x32,
        _wmma_gfx1250_b_16x16x32,
        _wmma_gfx12_row_col_16x16,
    ),
    # gfx1250 FP8/BF8 K=64 WMMA. A/B carry 32 low-bit bytes per lane presented
    # as <8 x i32>; accumulator is the same 16x16 column-distributed <8 x float>
    # as the f16/bf16 K=32 atom, so the accumulator map is shared with the f16
    # path. The A/B maps are slot-granular over the <8 x i32> carrier: slot i is
    # the dword holding K .. K+3, so a frag_len-slot walk spans 32 K-elements.
    # The block-scaled GEMM kernel predates these maps and still computes its
    # operand offsets directly; it is unaffected by their presence.
    "wmma_gfx1250_f32_16x16x64_fp8_fp8": _shared_src2_dst_frag_info(
        8,
        8,
        8,
        32,
        _wmma_gfx1250_a_16x16x64_f8,
        _wmma_gfx1250_b_16x16x64_f8,
        _wmma_gfx12_row_col_16x16,
    ),
    "wmma_gfx1250_f32_16x16x64_fp8_bf8": _shared_src2_dst_frag_info(
        8,
        8,
        8,
        32,
        _wmma_gfx1250_a_16x16x64_f8,
        _wmma_gfx1250_b_16x16x64_f8,
        _wmma_gfx12_row_col_16x16,
    ),
    "wmma_gfx1250_f32_16x16x64_bf8_fp8": _shared_src2_dst_frag_info(
        8,
        8,
        8,
        32,
        _wmma_gfx1250_a_16x16x64_f8,
        _wmma_gfx1250_b_16x16x64_f8,
        _wmma_gfx12_row_col_16x16,
    ),
    "wmma_gfx1250_f32_16x16x64_bf8_bf8": _shared_src2_dst_frag_info(
        8,
        8,
        8,
        32,
        _wmma_gfx1250_a_16x16x64_f8,
        _wmma_gfx1250_b_16x16x64_f8,
        _wmma_gfx12_row_col_16x16,
    ),
    # The same fp8 / bf8 pairs at K 128: 64 values of A and of B per lane,
    # counted in logical values; the accumulator map as at K 64. No A or B
    # lane map until one is checked against an independent source.
    **{
        f"wmma_gfx1250_f32_16x16x128_{a}_{b}": _shared_src2_dst_frag_info(
            64, 64, 8, 32, None, None, _wmma_gfx12_row_col_16x16
        )
        for a in ("fp8", "bf8")
        for b in ("fp8", "bf8")
    },
    # Except fp8 x fp8, which the gfx1250 fp8 GEMM runs on: it carries the
    # GEMM's A/B maps (see _wmma_gfx1250_a_16x16x128_f8), so like the K=64 rows
    # it counts the dword slots of its <16 x i32> carrier, 16 per lane.
    "wmma_gfx1250_f32_16x16x128_fp8_fp8": _shared_src2_dst_frag_info(
        16,
        16,
        8,
        32,
        _wmma_gfx1250_a_16x16x128_f8,
        _wmma_gfx1250_b_16x16x128_f8,
        _wmma_gfx12_row_col_16x16,
    ),
    "wmma_gfx1250_f32_16x16x32_bf16": _shared_src2_dst_frag_info(
        16,
        16,
        8,
        32,
        _wmma_gfx1250_a_16x16x32,
        _wmma_gfx1250_b_16x16x32,
        _wmma_gfx12_row_col_16x16,
    ),
    # bf16 x bf16 with an fp32 accumulator and a bf16 result: sixteen A and B
    # values and eight results per lane. No lane maps until one is checked
    # against an independent source.
    "wmma_gfx1250_bf16f32_16x16x32_bf16": _shared_src2_dst_frag_info(16, 16, 8, 32),
    # 16-bit results, fp8 / bf8 -> f16 and i8 -> i32, counted in logical
    # elements per lane: sixteen f16 / bf16 values of A and of B (K 32), 32 or
    # 64 fp8 / bf8 values (K 64 or K 128), 32 i8 values (K 64), and eight
    # results. No lane maps until one is checked against an independent source.
    "wmma_gfx1250_f16_16x16x32_f16": _shared_src2_dst_frag_info(16, 16, 8, 32),
    "wmma_gfx1250_bf16_16x16x32_bf16": _shared_src2_dst_frag_info(16, 16, 8, 32),
    **{
        f"wmma_gfx1250_f16_16x16x{k}_{a}_{b}": _shared_src2_dst_frag_info(
            k // 2, k // 2, 8, 32
        )
        for k in (64, 128)
        for a in ("fp8", "bf8")
        for b in ("fp8", "bf8")
    },
    **{
        op_id: _shared_src2_dst_frag_info(32, 32, 8, 32)
        for op_id in integer_mma_op_ids("wmma_gfx1250_i32_16x16x64_iu8", 8)
    },
    # --- WMMA f8f6f4 16x16x128 and f4 32x16x128 (wave32, gfx1250) -------------
    # Each format pair with a 6-bit or 4-bit operand: 64 values of A and of B
    # per lane in the f8f6f4 operand layout of its own format (16 words, two
    # groups of 16 lanes; CK's runs), and the gfx12 accumulator, the same as
    # the scaled f8f6f4 ops'. f4 32x16x128: 128 fp4 values of A and 64 of B per
    # lane, sixteen results; no lane maps until one is checked against an
    # independent source.
    **{
        op_id: _shared_src2_dst_frag_info(
            64,
            64,
            8,
            32,
            *_f8f6f4_operand_maps(a, b, 16, 2, 16),
            _wmma_gfx12_row_col_16x16,
        )
        for op_id, (a, b) in narrow_f8f6f4_op_ids(
            "wmma_gfx1250_f32_16x16x128_f8f6f4"
        ).items()
    },
    "wmma_gfx1250_f32_32x16x128_f4": _shared_src2_dst_frag_info(128, 64, 16, 32),
    # --- MFMA f8f6f4 16x16x128 and 32x32x64 (wave64, gfx950) ------------------
    # Each format pair with a 6-bit or 4-bit operand: 32 values of A and of B
    # per lane in the f8f6f4 operand layout of its own format (8 words; four
    # groups of 16 lanes at 16x16x128, two groups of 32 at 32x32x64, as the
    # public CDNA4 ISA operand tables give), and the f32 accumulator of the
    # shape.
    **{
        op_id: _shared_src2_dst_frag_info(
            32, 32, 4, 64, *_f8f6f4_operand_maps(a, b, 8, 4, 16), _mfma_row_col_16x16
        )
        for op_id, (a, b) in narrow_f8f6f4_op_ids("mfma_f32_16x16x128_f8f6f4").items()
    },
    **{
        op_id: _shared_src2_dst_frag_info(
            32, 32, 16, 64, *_f8f6f4_operand_maps(a, b, 8, 2, 32), _mfma_row_col_32x32
        )
        for op_id, (a, b) in narrow_f8f6f4_op_ids("mfma_f32_32x32x64_f8f6f4").items()
    },
    # The same instructions for each 8-bit x 8-bit format pair, fp8 x fp8 at
    # 16x16x128 under GFX950_FP8_16X16X128_OP_ID.
    **{
        op_id: _shared_src2_dst_frag_info(
            32, 32, 4, 64, *_f8f6f4_operand_maps(a, b, 8, 4, 16), _mfma_row_col_16x16
        )
        for op_id, (a, b) in gfx950_eight_bit_f8f6f4_op_ids((16, 16, 128)).items()
    },
    **{
        op_id: _shared_src2_dst_frag_info(
            32, 32, 16, 64, *_f8f6f4_operand_maps(a, b, 8, 2, 32), _mfma_row_col_32x32
        )
        for op_id, (a, b) in gfx950_eight_bit_f8f6f4_op_ids((32, 32, 64)).items()
    },
}


# Scaled operands share the padded <16 x i32> ABI, byte-carrier scales,
# and gfx12 accumulator map. Derive every ID from the expanded catalog of
# every arch that lists the family.
for _row in (row for spec in _load_specs().values() for row in spec["mma"]):
    if _row["family"] == "wmma_scaled":
        _scale_frag_len = 128 // _row["scale_block_k"]
        _MMA_FRAGMENT_INFO[_row["op_id"]] = _shared_src2_dst_frag_info(
            16,
            16,
            8,
            32,
            None,
            None,
            _wmma_gfx12_row_col_16x16,
            a_scale_frag_len=_scale_frag_len,
            b_scale_frag_len=_scale_frag_len,
            a_scale_fn=_wmma_gfx1250_a_scale,
            b_scale_fn=_wmma_gfx1250_b_scale,
        )
del _row, _scale_frag_len


def _mfma_scaled_a(builder, lane, slot):
    row = builder.mod(lane, builder.const_i32(32))
    k = builder.add(
        builder.mul(builder.div(lane, builder.const_i32(32)), builder.const_i32(32)),
        builder.const_i32(slot),
    )
    return row, k


def _mfma_scaled_f8_a(builder, lane, slot):
    row = builder.mod(lane, builder.const_i32(32))
    k = builder.add(
        builder.mul(builder.div(lane, builder.const_i32(32)), builder.const_i32(16)),
        builder.const_i32((slot // 16) * 32 + slot % 16),
    )
    return row, k


def _mfma_scaled_f8_b(builder, lane, slot):
    row, k = _mfma_scaled_f8_a(builder, lane, slot)
    return k, row


def _mfma_scaled_b(builder, lane, slot):
    row, k = _mfma_scaled_a(builder, lane, slot)
    return k, row


def _mfma_scaled_scale_a(builder, lane, slot):
    return builder.mod(lane, builder.const_i32(32)), builder.div(
        lane, builder.const_i32(32)
    )


def _mfma_scaled_scale_b(builder, lane, slot):
    row, k = _mfma_scaled_scale_a(builder, lane, slot)
    return k, row


_MMA_FRAGMENT_INFO.update(
    {
        f"mfma_gfx950_f32_32x32x64_{a}_{b}_scale_e8m0_e8m0_k32": _shared_src2_dst_frag_info(
            32,
            32,
            16,
            64,
            _mfma_scaled_f8_a if a in ("fp8", "bf8") else _mfma_scaled_a,
            _mfma_scaled_f8_b if b in ("fp8", "bf8") else _mfma_scaled_b,
            _mfma_row_col_32x32,
            1,
            1,
            _mfma_scaled_scale_a,
            _mfma_scaled_scale_b,
        )
        for a in ("fp8", "bf8", "fp6", "bf6", "fp4")
        for b in ("fp8", "bf8", "fp6", "bf6", "fp4")
    }
)


def _frag_info(op_id: str) -> _FragInfo:
    info = _MMA_FRAGMENT_INFO.get(op_id)
    if info is None:
        # An emulated atom has no fragment of its own: it borrows the one
        # belonging to the instruction it issues, so a bare logical op_id
        # resolves through its physical atom. One hop only -- the catalog
        # refuses an expansion chain (see _build_mma_emulation).
        physical = _op_id_physical_op_id().get(op_id)
        info = _MMA_FRAGMENT_INFO.get(physical) if physical else None
    if info is None:
        # Unknown atoms still load (e.g. a future JSON-only op_id); they carry
        # zero frag lengths and no maps until registered here.
        return _shared_src2_dst_frag_info(0, 0, 0, 64)
    return info


@dataclass(frozen=True)
class MemoryCapabilities:
    has_async_lds: bool
    has_async_global_lds: bool
    has_ds_read_tr: bool
    has_tdm: bool
    buffer_load_max_dwords: int
    # Widest per-lane DRAM->LDS DMA the ``buffer_load_lds`` family can do, in
    # dwords (0 where the arch has no such instruction). Distinct from
    # ``buffer_load_max_dwords``, which is the *register* vector buffer-load
    # width: CDNA3 loads a full dwordx4 into VGPRs but only a dword into LDS.
    async_lds_max_dwords: int
    # Feature flags. An arch sets one only after its feature passes a
    # functional run on a device; the lowerers admit the feature's operations
    # only on a target that sets it.
    has_scalar_data_prefetch: bool = False
    has_global_prefetch: bool = False
    has_cluster_launch: bool = False
    has_multicast_load: bool = False
    # Float add global atomics: scalar f32 ``global_atomic_add`` and the packed
    # ``<2 x bf16>`` / ``<2 x f16>`` ``global_atomic_pk_add``. Gates the WMMA
    # split-K wgrad epilogue; MFMA targets reach these atomics through their own
    # validated epilogues and do not consult it.
    has_global_float_atomic_add: bool = False


@dataclass(frozen=True)
class ResourceLimits:
    max_threads_per_block: int
    vgprs: int
    agprs: int
    sgprs: int


@dataclass(frozen=True)
class GemmCapabilities:
    """What the universal GEMM admits on an arch (``arch_specs.json`` "gemm").

    Unlike the hardware facts beside it, a flag here records that the GEMM
    path has been built and verified on the arch, so an arch whose hardware
    could run a feature still leaves it off until that has happened.
    """

    # WMMA warp-tile atoms (m, n, k) of the WMMA path; empty on MFMA targets.
    wmma_atoms: Tuple[Tuple[int, int, int], ...] = ()
    # fp8/bf8 A and B with a 16-bit C.
    fp8_ab: bool = False
    # The cshuffle epilogue on the WMMA path.
    wmma_cshuffle: bool = False
    # The cshuffle C tile reuses the A/B LDS, so the budget is max(AB, C)
    # rather than AB + C.
    cshuffle_lds_aliased: bool = False


def _prefer_native(ops: List[MmaOp]) -> List[MmaOp]:
    """Drop emulated atoms that a native atom of the same shape already covers.

    A native atom is always the better choice at equal shape: the emulated one
    costs several instructions to produce a result no more accurate. Filtering
    per shape rather than globally keeps an emulated shape the hardware has no
    native encoding for.

    Order is preserved, and a list with no emulated atoms is returned unchanged
    -- which is every query against the pre-XF32 catalog, so this cannot perturb
    existing selection.

    The key is the full shape ``(m, n, k)``, so a native atom of a *different* K
    does not shadow an emulated one -- and :meth:`MmaCatalog.select_largest_k`
    then prefers the larger K, which is the emulated atom whenever its K is the
    larger. That cannot happen today because no arch lists a native and an
    emulated atom for the same ``(a, b, c, m, n)``: gfx942 has only native TF32
    atoms and gfx950 only emulated ones, and
    ``test_no_arch_mixes_native_and_emulated_tf32`` pins it. Re-key this on
    ``(m, n)`` before an arch breaks that.
    """
    if all(op.is_native for op in ops):
        return ops
    native_shapes = {op.shape for op in ops if op.is_native}
    return [op for op in ops if op.is_native or op.shape not in native_shapes]


class MmaCatalog:
    """Query indexed matrix operands and an optional complete A/B scale contract.

    ``scales=None`` leaves scaling unconstrained; ``(None, None, None)``
    selects unscaled sources. Exact selection and largest-K ties must be unique.
    Legacy A/B/C queries default the destination dtype to the src2 dtype.
    """

    def __init__(self, ops: List[MmaOp]) -> None:
        self._ops = tuple(ops)

    @property
    def ops(self) -> Tuple[MmaOp, ...]:
        return self._ops

    def enumerate(
        self,
        *,
        family: str = "mma",
        src_dtypes: Optional[Tuple[str, str, str]] = None,
        dst_dtype: Optional[str] = None,
        scales: tuple[str | None, str | None, int | None] | None = None,
        a_dtype: Optional[str] = None,
        b_dtype: Optional[str] = None,
        c_dtype: Optional[str] = None,
        m: Optional[int] = None,
        n: Optional[int] = None,
    ) -> List[MmaOp]:
        if src_dtypes is None:
            if a_dtype is None or b_dtype is None or c_dtype is None:
                raise TypeError("enumerate requires src_dtypes or a/b/c_dtype")
            src_dtypes = (a_dtype, b_dtype, c_dtype)
        if scales is not None:
            if len(scales) != 3:
                raise ValueError("scales must contain exactly 3 entries")
            scales = _normalize_mma_scales(*scales)
        if len(src_dtypes) != 3:
            raise ValueError("src_dtypes must contain exactly 3 entries")
        src_keys = tuple(normalize_dtype(dtype) for dtype in src_dtypes)
        for i, legacy in enumerate((a_dtype, b_dtype, c_dtype)):
            if legacy is not None and normalize_dtype(legacy) != src_keys[i]:
                raise ValueError("conflicting indexed and legacy matrix dtypes")
        # Preserve the historical three-dtype query by defaulting dst to src2.
        # Indexed callers can state a distinct result.
        dst_key = normalize_dtype(src_dtypes[2] if dst_dtype is None else dst_dtype)
        out = []
        for op in self._ops:
            if op.family != family:
                continue
            if (
                tuple(src.dtype for src in op.srcs) != src_keys
                or op.dst.dtype != dst_key
            ):
                continue
            if scales is not None:
                a, b, block_k = scales
                expected = ((a, block_k), (b, block_k))
                actual = tuple(
                    (
                        (src.scale.dtype, src.scale.block_size)
                        if src.scale
                        else (None, None)
                    )
                    for src in op.srcs[:2]
                )
                if actual != expected:
                    continue
            if m is not None and op.m != m:
                continue
            if n is not None and op.n != n:
                continue
            out.append(op)
        return _prefer_native(out)

    def has_shape(
        self,
        *,
        family: str = "mma",
        src_dtypes: tuple[str, str, str] | None = None,
        a_dtype: str | None = None,
        b_dtype: str | None = None,
        c_dtype: str | None = None,
        dst_dtype: Optional[str] = None,
        scales: tuple[str | None, str | None, int | None] | None = None,
        m: int,
        n: int,
        k: int,
    ) -> bool:
        return any(
            op.shape == (m, n, k)
            for op in self.enumerate(
                family=family,
                src_dtypes=src_dtypes,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                c_dtype=c_dtype,
                dst_dtype=dst_dtype,
                scales=scales,
                m=m,
                n=n,
            )
        )

    def select_largest_k(
        self,
        *,
        family: str = "mma",
        src_dtypes: tuple[str, str, str] | None = None,
        a_dtype: str | None = None,
        b_dtype: str | None = None,
        c_dtype: str | None = None,
        dst_dtype: Optional[str] = None,
        scales: tuple[str | None, str | None, int | None] | None = None,
        m: int,
        n: int,
        k_max: Optional[int] = None,
    ) -> Optional[MmaOp]:
        cands = [
            op
            for op in self.enumerate(
                family=family,
                src_dtypes=src_dtypes,
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                c_dtype=c_dtype,
                dst_dtype=dst_dtype,
                scales=scales,
                m=m,
                n=n,
            )
            if k_max is None or op.k <= k_max
        ]
        if not cands:
            return None
        largest_k = max(op.k for op in cands)
        return self._unique([op for op in cands if op.k == largest_k])

    @staticmethod
    def _unique(ops: list[MmaOp]) -> MmaOp | None:
        if len(ops) > 1:
            raise ValueError("ambiguous MMA query; specify the full operand contract")
        return ops[0] if ops else None

    def by_op_id(self, op_id: str) -> Optional[MmaOp]:
        """Look up an atom by its ``op_id`` handle (the backend's MMA key)."""
        for op in self._ops:
            if op.op_id == op_id:
                return op
        return None

    def op_for_shape(
        self,
        *,
        family: str = "mma",
        src_dtypes: tuple[str, str, str] | None = None,
        a_dtype: str | None = None,
        b_dtype: str | None = None,
        c_dtype: str | None = None,
        dst_dtype: Optional[str] = None,
        scales: tuple[str | None, str | None, int | None] | None = None,
        m: int,
        n: int,
        k: int,
    ) -> Optional[MmaOp]:
        candidates = self.enumerate(
            family=family,
            src_dtypes=src_dtypes,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            dst_dtype=dst_dtype,
            scales=scales,
            m=m,
            n=n,
        )
        return self._unique([op for op in candidates if op.k == k])


@dataclass(frozen=True)
class ArchTarget:
    """Hardware-facts surface for one gfx target. Frozen; cheap to pass around."""

    gfx: str
    family: str
    target_family: str
    wave_size: int
    lds_capacity_bytes: int
    vmcnt_bits: int
    mma: MmaCatalog
    memory: MemoryCapabilities
    limits: ResourceLimits
    stepping: Optional[str] = None
    matrix_path: str = "mfma"
    has_mfma: bool = True
    has_wmma: bool = False
    waitcnt_model: str = "legacy"
    barrier_model: str = "legacy"
    requires_shader_end_padding: bool = False
    virtual_address_bits: int = 48
    wgp_cache_lds_shared: bool = False
    gemm: GemmCapabilities = GemmCapabilities()

    # --- identity ---------------------------------------------------------
    @property
    def isa_triple(self) -> str:
        return f"amdgcn-amd-amdhsa--{self.gfx}"

    # --- hardware predicates (policies compose these) ---------------------
    def fits_lds(self, bytes_in_use: int) -> bool:
        return bytes_in_use <= self.lds_capacity_bytes

    def supports_dtype_combo(
        self, a: str, b: str, c: str, *, family: str = "mma"
    ) -> bool:
        return (
            len(self.mma.enumerate(family=family, a_dtype=a, b_dtype=b, c_dtype=c)) > 0
        )

    def max_vector_load_dwords(self, dtype: str) -> int:
        # Width is gated by the buffer-load path, not the element type today.
        return self.memory.buffer_load_max_dwords

    @property
    def async_lds_max_dwords(self) -> int:
        """Widest per-lane DRAM->LDS DMA this arch can do, in dwords (0 = none).

        Every ``buffer_load_lds`` emitter must clamp to this. The backend does
        not diagnose an over-wide one -- it aborts the whole process.
        """
        return self.memory.async_lds_max_dwords

    @property
    def max_threads_per_block(self) -> int:
        return self.limits.max_threads_per_block

    @property
    def has_dpp_row_xmask(self) -> bool:
        """Whether DPP has ``row_xmask`` (an XOR with any mask 1..15 inside a
        16-lane row). It is a gfx10+ encoding: gfx11, gfx12 and gfx1250 have
        it; the gfx9 CDNA targets (``gfx9_mfma``, ``gfx950``) do not. An
        allow-list, so a new family gets the base-DPP encodings (valid on
        every target) until someone adds it here. Mirrored by
        ``rocke_arch_has_dpp_row_xmask`` in the C++ engine."""
        return self.target_family in ("gfx11_rdna", "gfx12_rdna", "gfx12_cdna")

    @staticmethod
    def from_gfx(gfx: str) -> "ArchTarget":
        return _build_target(gfx)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _op_id_dst_dtype() -> Dict[str, str]:
    """``op_id -> normalized dst dtype``, aggregated across every arch
    row in the JSON catalog.

    An ``op_id`` names a specific atom, so its ``dst`` dtype is invariant
    across the arches that list it. This lets a caller that only has a bare
    ``op_id`` string (e.g. :meth:`rocke.core.ir.IRBuilder.mma`) recover the
    ``dst`` dtype from the SSOT without holding an :class:`ArchTarget`.
    Op_ids that are frag-registered but not yet in the JSON catalog are absent
    (callers should treat a miss as the default f32 ``dst``).

    The **first** catalog hit for an op_id wins, matching the C implementation
    (``rocke_arch_mma_op_id_dst_dtype`` in ``query.cpp``, which returns the first
    registry hit). If a later arch lists the same op_id with a *different*
    ``dst`` dtype we raise instead of silently overwriting: that would be
    SSOT drift (the invariant above is broken) and must be fixed in the catalog,
    not masked. Raising here also keeps the Python and C engines deterministic
    and byte-identical rather than diverging on catalog ordering.
    """
    out: Dict[str, str] = {}
    for row in _load_specs().values():
        for o in row["mma"]:
            op_id = o["op_id"]
            _, dst_row = _mma_operand_rows(o)
            c = normalize_dtype(dst_row["dtype"])
            prev = out.get(op_id)
            if prev is None:
                out[op_id] = c  # first hit wins
            elif prev != c:
                raise ValueError(
                    f"arch SSOT drift in {_DATA_FILE.name}: op_id {op_id!r} has "
                    f"inconsistent dst dtype across arches "
                    f"({prev!r} vs {c!r}). An op_id names a specific atom, so its "
                    f"dst dtype must be invariant across the arches that "
                    f"list it; fix the catalog so every row agrees."
                )
    return out


@lru_cache(maxsize=1)
def _op_id_emulated_tf32_operands() -> Dict[str, Tuple[int, int, int]]:
    """``op_id -> (a_frag_len, b_frag_len, c_frag_len)`` for each emulated TF32 atom.

    The operand contract the verifier enforces for an atom the hardware does not
    encode: A and B are ``vec<tf32xN>``, N being the fragment length the atom
    borrows from the physical atom it expands into, and the accumulator is
    ``vec<f32xM>``. Read from the catalog rather than restated, so a new emulated
    row needs no verifier edit. Same first-hit and raise-on-drift contract as
    :func:`_op_id_dst_dtype` -- an op_id names a specific atom, so its fragments
    must be invariant across the arches that list it. Mirrored by
    ``rocke_arch_mma_op_id_emulated_operands`` in ``query.cpp``.
    """
    out: Dict[str, Tuple[int, int, int]] = {}
    for row in _load_specs().values():
        for o in row["mma"]:
            emulation = o.get("emulation")
            if emulation is None:
                continue
            srcs, _ = _mma_operand_rows(o)
            if tuple(normalize_dtype(src["dtype"]) for src in srcs[:2]) != (
                "tf32",
                "tf32",
            ):
                continue
            info = _frag_info(emulation["physical_op_id"])
            lens = (info.srcs[0].frag_len, info.srcs[1].frag_len, info.srcs[2].frag_len)
            if not all(lens):
                # _frag_info answers zeros for an op_id it does not know, and the
                # verifier would then demand vec<tf32x0>. Refuse here, where the
                # catalog row is still in view, rather than at a confusing use site.
                raise ValueError(
                    f"emulated atom {o['op_id']!r} expands into "
                    f"{emulation['physical_op_id']!r}, which has no fragment metadata"
                )
            prev = out.setdefault(o["op_id"], lens)  # first hit wins
            if prev != lens:
                raise ValueError(
                    f"arch SSOT drift in {_DATA_FILE.name}: op_id {o['op_id']!r} "
                    f"has inconsistent operand fragments across arches "
                    f"({prev!r} vs {lens!r}). An op_id names a specific atom, so "
                    f"its fragments must be invariant across the arches that list "
                    f"it; fix the catalog so every row agrees."
                )
    return out


def emulated_tf32_operand_lens(op_id: str) -> Optional[Tuple[int, int, int]]:
    """Operand fragment lengths ``(a, b, c)`` if ``op_id`` is an emulated TF32 atom.

    ``None`` for every atom the hardware encodes, which is the answer for every
    ordinary MMA, so the verifier's hot path is one dict miss.
    """
    return _op_id_emulated_tf32_operands().get(op_id)


@lru_cache(maxsize=1)
def _op_id_physical_op_id() -> Dict[str, str]:
    """``op_id -> physical op_id`` for every emulated atom in the JSON catalog.

    The bare-op_id twin of ``MmaOp.emulation.physical_op_id``. A caller holding
    only a logical op_id string (e.g. :meth:`rocke.core.ir.IRBuilder.mma`) needs
    it to size a fragment the emulated atom borrows rather than owns. Natively
    encoded rows contribute nothing. Same first-hit and raise-on-drift contract
    as :func:`_op_id_dst_dtype` — an op_id names a specific atom, so the
    instruction it expands into must be invariant across the arches listing it.
    Mirrored by ``rocke_arch_mma_op_id_physical_op_id`` in ``query.cpp``.
    """
    out: Dict[str, str] = {}
    for row in _load_specs().values():
        for o in row["mma"]:
            emulation = o.get("emulation")
            if emulation is None:
                continue
            op_id, physical = o["op_id"], emulation["physical_op_id"]
            prev = out.setdefault(op_id, physical)  # first hit wins
            if prev != physical:
                raise ValueError(
                    f"arch SSOT drift in {_DATA_FILE.name}: op_id {op_id!r} "
                    f"expands into different physical atoms across arches "
                    f"({prev!r} vs {physical!r}). An op_id names a specific atom, "
                    f"so its expansion must be invariant across the arches that "
                    f"list it; fix the catalog so every row agrees."
                )
    return out


def _mma_op_by_id(op_id: str) -> MmaOp | None:
    """The catalog atom named ``op_id``, or ``None`` when no arch lists it.

    For a caller that holds only an op_id, such as the operand contract of a
    scaled atom, which does not depend on the target lowering it. An op_id
    names one atom, so every arch that lists it holds the same row; the first
    hit in :func:`known_arches` order is returned. Mirrored by
    ``rocke_arch_mma_op_by_id`` in ``query.cpp``.
    """
    for gfx in known_arches():
        atom = _build_target(gfx).mma.by_op_id(op_id)
        if atom is not None:
            return atom
    return None


@lru_cache(maxsize=1)
def _op_id_family() -> Dict[str, str]:
    """Resolve operation families without selecting a target or parsing IDs."""
    out: Dict[str, str] = {}
    for row in _load_specs().values():
        for op in row["mma"]:
            op_id, family = op["op_id"], op["family"]
            previous = out.setdefault(op_id, family)
            if previous != family:
                raise ValueError(
                    f"arch SSOT drift: op_id {op_id!r} has inconsistent family "
                    f"across arches ({previous!r} vs {family!r})"
                )
    return out


def _mma_operand_rows(o: dict) -> tuple[list[dict], dict]:
    """Validate duplicate and structural matrix metadata before either reader.

    Legacy JSON C names src2; only an omitted dst defaults to that source.
    Destination format support is checked separately when IR is built.
    """
    op_id = o["op_id"]

    def matrix_row(row: object, role: str) -> dict:
        if not isinstance(row, dict) or not isinstance(row.get("dtype"), str):
            raise ValueError(
                f"MMA catalog op {op_id!r} {role} must define a string dtype"
            )
        return dict(row)

    if "srcs" in o:
        rows = o["srcs"]
        if not isinstance(rows, (list, tuple)) or len(rows) != 3:
            raise ValueError(
                f"MMA catalog op {op_id!r} must define exactly 3 matrix sources"
            )
        src_rows = [matrix_row(row, f"src{i}") for i, row in enumerate(rows)]
        for i, legacy in enumerate(("a", "b", "c")):
            if legacy in o:
                legacy_row = matrix_row({"dtype": o[legacy]}, legacy)
                if normalize_dtype(legacy_row["dtype"]) != normalize_dtype(
                    src_rows[i]["dtype"]
                ):
                    raise ValueError("conflicting indexed and legacy matrix dtypes")
    else:
        src_rows = [matrix_row({"dtype": o.get(key)}, key) for key in ("a", "b", "c")]
    destination = o["dst"] if "dst" in o else src_rows[2]
    # Published legacy rows may spell a distinct result as a dtype string.
    if isinstance(destination, str):
        destination = {"dtype": destination}
    dst_row = matrix_row(destination, "dst")
    if "a_scale_dtype" in o or "b_scale_dtype" in o or "scale_block_k" in o:
        a, b, block_k = _normalize_mma_scales(
            o.get("a_scale_dtype"), o.get("b_scale_dtype"), o.get("scale_block_k")
        )
        src_rows = [dict(row) for row in src_rows]
        for i, dtype in enumerate((a, b)):
            legacy_scale = (
                MmaScaleOperand(dtype, block_k) if dtype is not None else None
            )
            if "scale" in src_rows[i]:
                scale = src_rows[i]["scale"]
                indexed_scale = MmaScaleOperand(**scale) if scale is not None else None
                if indexed_scale != legacy_scale:
                    raise ValueError("conflicting indexed and legacy scale metadata")
            if legacy_scale is not None:
                src_rows[i]["scale"] = {"dtype": dtype, "block_size": block_k}
    return src_rows, dst_row


def _build_mma_op(o: dict) -> MmaOp:
    """Construct an :class:`MmaOp` from one catalog JSON row, attaching the
    physical fragment lengths and layout maps registered for its op_id."""
    op_id = o["op_id"]
    emulation = _build_mma_emulation(o.get("emulation"))
    # An emulated atom has no fragment layout of its own: it borrows the one
    # belonging to the instruction it actually issues. Deriving here (rather
    # than duplicating a row in _MMA_FRAGMENT_INFO) is what makes it impossible
    # for the logical atom and its physical atom to drift apart.
    info = _frag_info(emulation.physical_op_id if emulation else op_id)

    def _mk(role: str, frag_len: int, fn: _LaneCoordFn) -> Optional[LayoutMap]:
        if fn is None or frag_len <= 0:
            return None
        return LayoutMap(role=role, frag_len=frag_len, wave_size=info.wave_size, fn=fn)

    src_rows, dst_row = _mma_operand_rows(o)
    if emulation is not None:
        a_dtype, b_dtype = (normalize_dtype(row["dtype"]) for row in src_rows[:2])
        if (
            a_dtype != emulation.recipe.operand_dtype
            or b_dtype != emulation.recipe.operand_dtype
        ):
            raise ValueError(
                f"MMA emulation recipe {str(emulation.recipe)!r} emulates "
                f"{emulation.recipe.operand_dtype} operands; row {op_id!r} has "
                f"{a_dtype}/{b_dtype}"
            )

    def _source(
        row: dict,
        role: str,
        frag_len: int,
        fn: _LaneCoordFn,
        scale_len: int = 0,
        scale_fn: _LaneCoordFn = None,
    ) -> MmaSrc:
        scale_row = row.get("scale")
        scale = (
            MmaScaleOperand(
                dtype=scale_row["dtype"],
                block_size=scale_row["block_size"],
                frag_len=scale_len,
                layout=_mk(f"scale_{role}", scale_len, scale_fn),
            )
            if scale_row is not None
            else None
        )
        return MmaSrc(
            dtype=normalize_dtype(row["dtype"]),
            frag_len=frag_len,
            layout=_mk(role, frag_len, fn),
            scale=scale,
        )

    return MmaOp(
        family=o["family"],
        srcs=(
            _source(
                src_rows[0],
                "src0",
                info.srcs[0].frag_len,
                info.srcs[0].fn,
                info.a_scale_frag_len,
                info.a_scale_fn,
            ),
            _source(
                src_rows[1],
                "src1",
                info.srcs[1].frag_len,
                info.srcs[1].fn,
                info.b_scale_frag_len,
                info.b_scale_fn,
            ),
            _source(src_rows[2], "src2", info.srcs[2].frag_len, info.srcs[2].fn),
        ),
        dst=MmaDst(
            dtype=normalize_dtype(dst_row["dtype"]),
            frag_len=info.dst.frag_len,
            layout=_mk("dst", info.dst.frag_len, info.dst.fn),
        ),
        m=o["m"],
        n=o["n"],
        k=o["k"],
        op_id=op_id,
        wave_size=info.wave_size,
        emulation=emulation,
    )


def _build_mma_emulation(spec: Optional[dict]) -> Optional[MmaEmulation]:
    """Build the emulation descriptor for one catalog row, if it has one.

    An absent ``emulation`` key means the atom is natively encoded, which is
    every atom the catalog shipped before XF32.
    """
    if spec is None:
        return None
    unknown = set(spec) - {"recipe", "physical_op_id"}
    if unknown:
        raise ValueError(f"unknown MMA emulation keys: {sorted(unknown)}")
    try:
        spelling, physical_op_id = spec["recipe"], spec["physical_op_id"]
    except KeyError as exc:
        raise ValueError(f"MMA emulation is missing required key {exc}") from None
    # Only the enum lookup is "an unknown recipe". Parsing it on its own keeps the
    # descriptor's other checks, such as an empty physical_op_id, reporting what
    # they actually found instead of blaming a recipe that is perfectly valid.
    try:
        recipe = MmaEmulationRecipe(spelling)
    except ValueError:
        raise ValueError(
            f"unknown MMA emulation recipe {spelling!r}; expected one of "
            f"{[str(r) for r in MmaEmulationRecipe]}"
        ) from None
    emulation = MmaEmulation(recipe=recipe, physical_op_id=physical_op_id)
    # The physical atom must be frag-registered, or the emulated row would
    # silently inherit the zero-length fallback and produce wrong vector widths.
    if emulation.physical_op_id not in _MMA_FRAGMENT_INFO:
        raise ValueError(
            f"MMA emulation physical_op_id {emulation.physical_op_id!r} is not "
            f"registered in _MMA_FRAGMENT_INFO; add its fragment metadata first"
        )
    return emulation


@lru_cache(maxsize=None)
def _build_target(gfx: str) -> ArchTarget:
    specs = _load_specs()
    if gfx not in specs:
        raise KeyError(
            f"unknown gfx target {gfx!r}; known: {sorted(specs)}. "
            f"Add a row to {_DATA_FILE.name}."
        )
    row = specs[gfx]
    mma = MmaCatalog([_build_mma_op(o) for o in row["mma"]])
    for op in mma.ops:
        if op.emulation is None:
            continue
        # The physical atom is checked against _MMA_FRAGMENT_INFO at row load,
        # which is global. Whether *this* arch can issue it is a per-arch
        # question, and answering it wrongly makes the catalog claim support the
        # compiler then rejects ("Cannot select") at codegen.
        physical = mma.by_op_id(op.emulation.physical_op_id)
        if physical is None or not physical.is_native:
            raise ValueError(
                f"{gfx}: emulated atom {op.op_id!r} expands into "
                f"{op.emulation.physical_op_id!r}, which is not a natively "
                f"encoded atom of {gfx}"
            )
    mem = row["memory"]
    lim = row["limits"]
    has_mfma = any(op.family == "mma" for op in mma.ops)
    has_wmma = any(op.family == "wmma" for op in mma.ops)
    return ArchTarget(
        gfx=gfx,
        family=row["family"],
        target_family=row["target_family"],
        wave_size=row["wave_size"],
        lds_capacity_bytes=row["lds_capacity_bytes"],
        vmcnt_bits=row["vmcnt_bits"],
        mma=mma,
        memory=MemoryCapabilities(
            has_async_lds=mem["has_async_lds"],
            has_async_global_lds=mem.get("has_async_global_lds", mem["has_async_lds"]),
            has_ds_read_tr=mem["has_ds_read_tr"],
            has_tdm=mem.get("has_tdm", False),
            buffer_load_max_dwords=mem["buffer_load_max_dwords"],
            async_lds_max_dwords=mem["async_lds_max_dwords"],
            has_scalar_data_prefetch=mem.get("has_scalar_data_prefetch", False),
            has_global_prefetch=mem.get("has_global_prefetch", False),
            has_cluster_launch=mem.get("has_cluster_launch", False),
            has_multicast_load=mem.get("has_multicast_load", False),
            has_global_float_atomic_add=mem.get("has_global_float_atomic_add", False),
        ),
        limits=ResourceLimits(
            max_threads_per_block=lim["max_threads_per_block"],
            vgprs=lim["vgprs"],
            agprs=lim["agprs"],
            sgprs=lim["sgprs"],
        ),
        stepping=row.get("stepping"),
        matrix_path=row.get(
            "matrix_path",
            "wmma" if has_wmma and not has_mfma else "mfma" if has_mfma else "scalar",
        ),
        has_mfma=row.get("has_mfma", has_mfma),
        has_wmma=row.get("has_wmma", has_wmma),
        waitcnt_model=row.get("waitcnt_model", "legacy"),
        barrier_model=row.get("barrier_model", "legacy"),
        requires_shader_end_padding=row.get("requires_shader_end_padding", False),
        virtual_address_bits=row.get("virtual_address_bits", 48),
        wgp_cache_lds_shared=row.get("wgp_cache_lds_shared", False),
        gemm=_gemm_capabilities(row.get("gemm", {})),
    )


def _gemm_capabilities(row: dict) -> GemmCapabilities:
    return GemmCapabilities(
        wmma_atoms=tuple(tuple(atom) for atom in row.get("wmma_atoms", ())),
        fp8_ab=row.get("fp8_ab", False),
        wmma_cshuffle=row.get("wmma_cshuffle", False),
        cshuffle_lds_aliased=row.get("cshuffle_lds_aliased", False),
    )


def known_arches() -> Tuple[str, ...]:
    return tuple(sorted(_load_specs()))


def arches_with_own_facts() -> tuple[str, ...]:
    """Every known arch whose ``arch_specs.json`` row holds its own facts, sorted.

    A row that takes its facts from a generic target (``facts_of``) is left out:
    its target is the generic's under another name, and :func:`listed_arch_for`
    admits it wherever the generic is listed. Every known arch is therefore
    either in this tuple or covered by an entry of it, and each distinct set of
    facts appears once. Declare these as a candidate's arches, or iterate them,
    when the claim is "every target": iterating :func:`known_arches` instead
    repeats each generic's work once per member that points at it.
    """
    pointers = _load_pointers()
    return tuple(gfx for gfx in known_arches() if gfx not in pointers)


def target_id_from_isa(isa: str) -> str:
    """Extract the target ID from a COMGR ISA name.

    ``compile_kernel(..., isa=...)`` passes its ``isa`` argument here. That
    value may come from an example's ``--isa`` option, a fixed string in a
    script, or the compile helper's ``gfx950`` default. See the input paths
    documented in :mod:`rocke.helpers.compile`.

    For ``amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-``, this returns
    ``gfx942:sramecc+:xnack-``, keeping any profile or feature suffix.
    It also accepts a target ID without the ISA prefix.

    The result starts at the last ``gfx`` in the input. If there is no
    ``gfx``, the input is returned unchanged. This does not validate the name;
    :func:`parse_target_id` does.
    """

    start = isa.rfind("gfx")
    return isa[start:] if start >= 0 else isa


# The whole of a value parse_target_id accepts: an optional AMDHSA triple with an
# empty or "unknown" environment field, a processor, then feature suffixes.
_TARGET_ID = re.compile(
    r"(?:amdgcn-amd-amdhsa-(?:unknown)?-)?"
    r"(gfx[0-9a-z]+(?:-[0-9a-z]+)*(?::[0-9a-z_]+[+-])*)"
)


def parse_target_id(value: str) -> str:
    """Return the target ID that ``value`` names, refusing any other form.

    ``value`` is either an AMDHSA ISA name, as a code object's ``amdhsa.target``
    note records it, or a bare target ID. All of it must be, in order:

    - optionally the triple ``amdgcn-amd-amdhsa-`` with an empty or ``unknown``
      environment field and its closing ``-`` (``amdgcn-amd-amdhsa--`` or
      ``amdgcn-amd-amdhsa-unknown-``; compilers write either);
    - a processor: ``gfx`` then lower-case letters and digits, with ``-``
      separated parts for names such as ``gfx11-generic`` or ``gfx9-4-generic``;
    - zero or more feature suffixes, each ``:`` then a name then ``+`` or ``-``
      (``:sramecc+:xnack-``).

    Returns the target ID with its feature suffixes, without the triple:
    ``amdgcn-amd-amdhsa-unknown-gfx942:xnack-`` gives ``gfx942:xnack-``.
    :func:`processor_from_target_id` then gives the processor.

    Raises ``ValueError`` naming the value for anything else: an empty value, an
    other environment field (``amdgcn-amd-amdhsa-linux-gfx950``), a processor
    name in the wrong place (``notgfx950``, ``amdgcn-amd-amdhsa-unknown-:gfx950``)
    or a feature without its sign (``gfx950:xnack``). Unlike
    :func:`target_id_from_isa`, it never guesses a processor out of a malformed
    value. It does not check that rocKE, the compiler or a device supports the
    target.
    """

    match = _TARGET_ID.fullmatch(value)
    if match is None:
        # The value is quoted by hand, not with !r, so rocke_parse_target_id in
        # the C++ engine writes the same text byte for byte.
        raise ValueError(
            f"'{value}' is not a target ID: expected an optional "
            "amdgcn-amd-amdhsa-- or amdgcn-amd-amdhsa-unknown- prefix, a processor "
            "such as gfx942 or gfx11-generic, then any :feature+ or :feature- "
            "suffixes"
        )
    return match.group(1)


def processor_from_target_id(target_id: str) -> str:
    """Return the processor of a target ID, without profile or feature suffixes.

    ``gfx942:sramecc+:xnack-`` gives ``gfx942`` and ``gfx1250-strict`` gives
    ``gfx1250``; catalog names such as ``gfx11-generic`` stay whole, and a
    member of a generic target stays itself (``gfx1100`` stays ``gfx1100``;
    :func:`generic_arch_from_target_id` gives its generic). Any other name is
    cut to its leading ``gfx`` token of lower-case letters and digits; a name
    that does not start with ``gfx`` is returned up to its first ``:``. This
    does not validate the name; validate it first with :func:`parse_target_id`
    when it comes from outside rocKE.
    """

    target_without_features = target_id.split(":", 1)[0]
    arches = known_arches()
    if target_without_features in arches:
        return target_without_features
    for arch in sorted(arches, key=len, reverse=True):
        if target_without_features.startswith(f"{arch}-"):
            return arch
    for members in _load_generic_targets().values():
        for processor in members:
            if (
                target_without_features == processor
                or target_without_features.startswith(f"{processor}-")
            ):
                return processor
    match = re.match(r"^(gfx[0-9a-z]+)", target_without_features)
    return match.group(1) if match else target_without_features


def base_arch_from_target_id(target_id: str) -> str:
    """Derive the architecture name used for rocKE catalog lookup and lowering.

    Removes features after ``:`` and profile suffixes such as ``-strict``:
    ``gfx1250-strict`` becomes ``gfx1250`` and ``gfx942:xnack-`` becomes
    ``gfx942``. Names already in :func:`known_arches`, including
    ``gfx11-generic`` and every member of a catalogued generic target
    (``gfx1100``, ``gfx1200``), are preserved. A processor in one of LLVM's
    documented generic compatibility domains (``generic_targets``) that has no
    row of its own uses that generic rocKE catalog row.

    Any other unknown name is reduced to its leading ``gfx`` token when
    possible. This does not check support; :meth:`ArchTarget.from_gfx` requires
    a matching catalog entry.
    """

    target_without_features = target_id.split(":", 1)[0]
    processor = processor_from_target_id(target_id)
    if processor in known_arches():
        return processor
    for generic, members in _load_generic_targets().items():
        if processor in members and (
            target_without_features == processor
            or target_without_features.startswith(f"{processor}-")
        ):
            return generic
    return processor


def compiler_target_from_target_id(target_id: str) -> str:
    """Derive the compiler target name for ``target_id``.

    Removes profile suffixes such as ``-strict`` from the processor token, but
    keeps features after ``:``. Catalog selection remains independent, so a
    concrete processor may compile for itself while lowering through a generic
    rocKE catalog row. For example, ``gfx1250-strict`` becomes ``gfx1250``,
    while ``gfx942:sramecc+:xnack-`` stays unchanged.

    This only converts the string. :func:`compiler_target_for_flavor` decides
    which name a given compiler is passed.
    """

    target_without_features, separator, features = target_id.partition(":")
    processor = processor_from_target_id(target_id)
    if target_without_features.startswith(f"{processor}-"):
        target_without_features = processor
    if separator:
        return f"{target_without_features}:{features}"
    return target_without_features


def compiler_target_for_flavor(target_id: str, *, llvm_flavor: str) -> str:
    """The compiler target to pass to the ``llvm_flavor`` compiler for ``target_id``.

    Starts from :func:`compiler_target_from_target_id` (profile removed,
    features kept) and checks its processor against what the ``llvm_flavor``
    compiler was measured to accept (``processors`` in that flavor's
    arch-domain column):

    * accepted, or never measured (no committed column, or a name with no
      ``arch_specs.json`` row): the processor itself, and the compiler has the
      last word;
    * not accepted, and its row points at a generic target (``facts_of``) the
      compiler accepts: that generic target, whose code objects load on the
      processor, with a ``UserWarning`` that says so (Python's default warning
      filter shows it once per process for each processor and flavor);
    * not accepted otherwise: ``ValueError``.

    Features after ``:`` are kept either way. Catalog selection is independent
    of this name: the processor lowers through its own row whichever name it is
    compiled for. Pass the flavor of the compiler that actually runs.
    """
    processor, separator, features = compiler_target_from_target_id(
        target_id
    ).partition(":")
    name = _processor_the_compiler_accepts(processor, llvm_flavor)
    return f"{name}{separator}{features}"


def compiler_target_for_llvm_major(target_id: str, llvm_major: int | None) -> str:
    """The compiler target to pass to the LLVM ``llvm_major`` compiler for ``target_id``.

    Pass the major of the compiler that will run (the one COMGR loads, or
    hipcc's clang), never an emission override such as ``ROCKE_LLVM_FLAVOR``.
    A major in a qualified lane routes through that lane's measured processor
    answers (:func:`compiler_target_for_flavor`). A major newer than every
    lane, or ``None`` for a compiler whose version is unknown, was never
    measured: the processor keeps its name
    (:func:`compiler_target_from_target_id`), and whether that compiler takes
    it is left to the compiler.
    """
    flavor = None if llvm_major is None else measured_flavor_for_llvm(llvm_major)
    if flavor is None:
        return compiler_target_from_target_id(target_id)
    return compiler_target_for_flavor(target_id, llvm_flavor=flavor)


def _processor_the_compiler_accepts(processor: str, llvm_flavor: str) -> str:
    """``processor``, or the generic target to compile it as (see
    :func:`compiler_target_for_flavor`)."""
    column = arch_domain.load(llvm_flavor)
    if column is None or column.accepts_processor(processor) is not False:
        return processor
    generic = _load_pointers().get(processor)
    if generic is not None and column.accepts_processor(generic) is not False:
        warnings.warn(
            f"the {llvm_flavor} compiler does not accept the processor "
            f"{processor!r}; compiling for {generic!r} instead, the generic "
            f"target its arch_specs.json row points at, whose code objects load "
            f"on {processor!r}",
            stacklevel=3,
        )
        return generic
    raise ValueError(
        f"the {llvm_flavor} compiler does not accept the processor {processor!r}, "
        f"and its arch_specs.json row points at no generic target that compiler "
        f"accepts -- compile with an LLVM flavor whose compiler accepts {processor!r}"
    )


def generic_arch_from_target_id(target_id: str) -> Optional[str]:
    """Return the catalogued LLVM generic target that runs on ``target_id``.

    A code object built for the returned generic processor loads on the device
    named by ``target_id``. ``gfx1100`` and ``gfx1151:xnack-`` both become
    ``gfx11-generic``, and a generic name maps to itself. Returns ``None`` when
    no catalogued generic target covers the processor.

    This only converts the string; it never queries a GPU. Unlike
    :func:`base_arch_from_target_id`, it deliberately replaces a concrete name,
    so call it only where one generic code object should serve the family.
    """

    base_arch = base_arch_from_target_id(target_id)
    for generic, members in _load_generic_targets().items():
        if base_arch == generic or base_arch in members:
            return generic
    return None


def listed_arch_for(arch: str, listed: Iterable[str]) -> Optional[str]:
    """The entry of ``listed`` that covers ``arch``, or ``None`` when none does.

    That is ``arch`` itself when it is listed, else the generic target its
    ``arch_specs.json`` row points at (``facts_of``) when that is listed. A list
    that names ``gfx11-generic`` therefore covers every gfx11 processor whose row
    points there, and a processor the list names still matches itself first.
    A row that holds its own facts, and a name with no row, match only
    themselves.

    Use the returned entry, not ``arch``, to look up anything the list's owner
    keyed by its entries (a per-target table, a candidate name).
    """
    entries = tuple(listed)
    if arch in entries:
        return arch
    generic = _load_pointers().get(arch)
    return generic if generic in entries else None


def arch_from_isa(isa: str) -> str:
    """Extract a target ID from a COMGR ISA name, then derive its base architecture.

    Combines :func:`target_id_from_isa` and :func:`base_arch_from_target_id`.
    For example, ``amdgcn-amd-amdhsa--gfx942:xnack-`` becomes ``gfx942``.
    """

    return base_arch_from_target_id(target_id_from_isa(isa))


def validate_arch(arch: Optional[str]) -> None:
    if arch is None:
        raise ValueError(
            "Could not detect a GPU architecture. Pass in an explicit architecture instead."
        )
    if arch not in known_arches():
        raise ValueError(
            f"Unknown GPU architecture detected. Known architectures include: {known_arches()}"
        )
