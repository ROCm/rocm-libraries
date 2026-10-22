# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One LDS arena with phase overlays, for persistent layer kernels.

A layer kernel runs many stages, each with its own LDS needs; stages of
different *phases* never hold LDS at the same time, so their entries overlay
each other in one allocation. The layout rules are those of the ATOM mono
layer kernel (``layout._place``):

* every entry is rounded up to 4 words (16 bytes);
* a ``per_row`` entry of a ``rows > 1`` build holds one copy per request row,
  rows ``stride`` words apart, with the stride padded to 16 words mod 64 so
  the rows' copies of a word sit in distinct banks;
* every phase starts at word 0; the arena is the largest phase, plus the
  ``pinned`` entries (live for the whole kernel, e.g. timeline stamps) after
  it.

The arena is one ``smem_alloc(I32, [words])``: offsets are explicit, so the
overlay lifetimes are visible in the code rather than left to the lowering's
liveness aliasing. ``min_words`` pads the allocation, e.g. above half of a
CU's LDS so that exactly one CTA fits per CU (the co-residency argument of a
resident grid launched without cooperative launch).

A stage changing phase must be preceded by a CTA barrier with a compiler
fence (``s_barrier_bare(fence="compiler")`` or ``sync_lds_only``):
:meth:`rocke.helpers.task_schedule.TaskTable.lds_barriers` says where.
Emission uses only ``tile.smem_*`` ops (both engines).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rocke.core.ir import F32, I32, IRBuilder, Value

__all__ = [
    "CU_LDS_WORDS_GFX950",
    "LDS_LIMIT_WORDS",
    "LdsArena",
    "LdsEntry",
    "LdsView",
    "ONE_CTA_PER_CU_WORDS",
]

# gfx950: 160 KiB of LDS per CU; one workgroup may allocate all of it.
CU_LDS_WORDS_GFX950 = 40960
LDS_LIMIT_WORDS = CU_LDS_WORDS_GFX950
# Above half a CU's LDS: a second CTA never fits on the CU.
ONE_CTA_PER_CU_WORDS = CU_LDS_WORDS_GFX950 // 2 + 64


@dataclass(frozen=True)
class LdsEntry:
    """``words`` i32 words; ``per_row``: one copy per request row."""

    name: str
    words: int
    per_row: bool = False

    def __post_init__(self) -> None:
        if self.words < 1:
            raise ValueError(f"{self.name}: words must be >= 1")


def _entry(e: LdsEntry | tuple) -> LdsEntry:
    return e if isinstance(e, LdsEntry) else LdsEntry(*e)


class LdsArena:
    """Offsets of every entry of every phase (static, host side).

    ``phases``: ``{phase: [LdsEntry | (name, words[, per_row])]}``, entries in
    placement order. ``pinned``: entries outside every phase, placed after
    the largest phase. Names are unique across all phases.
    """

    def __init__(
        self,
        phases: Mapping[str, Sequence[LdsEntry | tuple]],
        *,
        rows: int = 1,
        pinned: Sequence[LdsEntry | tuple] = (),
        limit: int = LDS_LIMIT_WORDS,
        min_words: int = 0,
    ):
        if rows < 1:
            raise ValueError("rows must be >= 1")
        self.rows = rows
        self.offset: dict[str, int] = {}
        self.stride: dict[str, int] = {}
        self.phase_of: dict[str, str | None] = {}
        self.phase_words: dict[str, int] = {}
        for ph, entries in phases.items():
            self.phase_words[ph] = self._place([_entry(e) for e in entries], 0, ph)
        top = max(self.phase_words.values(), default=0)
        end = self._place([_entry(e) for e in pinned], top, None)
        self.used_words = end
        self.words = max(end, min_words)
        self.words = (self.words + 3) // 4 * 4
        if self.words > limit:
            raise ValueError(
                f"LDS arena of {self.words} words exceeds the limit of {limit}"
            )

    def _place(self, entries: list[LdsEntry], w: int, phase: str | None) -> int:
        for e in entries:
            if e.name in self.offset:
                raise ValueError(f"duplicate LDS entry {e.name!r}")
            st = (e.words + 3) // 4 * 4
            if self.rows > 1 and e.per_row:
                st += (16 - st) % 64
            self.offset[e.name] = w
            self.stride[e.name] = st
            self.phase_of[e.name] = phase
            w += st * (self.rows if e.per_row else 1)
        return w

    def at(self, name: str, row: int = 0) -> int:
        """Static word offset of ``name`` in request row ``row``."""
        return self.offset[name] + row * self.stride[name]

    def one_cta_per_cu(self, cu_words: int = CU_LDS_WORDS_GFX950) -> bool:
        return 2 * self.words > cu_words

    def emit(self, b: IRBuilder, name_hint: str = "arena") -> LdsView:
        """The arena's one ``smem_alloc`` and a view over it."""
        return LdsView(self, b.smem_alloc(I32, [self.words], name_hint=name_hint))


class LdsView:
    """Typed loads and stores into an emitted :class:`LdsArena`.

    ``idx`` is a word index inside the entry (``int`` or i32 ``Value``);
    ``row`` a request row (``int`` or uniform i32 ``Value``).
    """

    def __init__(self, arena: LdsArena, smem: Value):
        self.arena = arena
        self.smem = smem

    def word(self, b: IRBuilder, name: str, idx: int | Value = 0, row=0) -> Value:
        """The arena word index of ``name[idx]`` in row ``row`` (i32)."""
        a = self.arena
        if isinstance(row, int) and isinstance(idx, int):
            return b.const_i32(a.at(name, row) + idx)
        off = b.const_i32(a.offset[name])
        if not (isinstance(row, int) and row == 0):
            r = b.const_i32(row) if isinstance(row, int) else row
            off = b.add(off, b.mul(r, b.const_i32(a.stride[name])))
        i = b.const_i32(idx) if isinstance(idx, int) else idx
        return b.add(off, i)

    def ld_i32(self, b: IRBuilder, name: str, idx: int | Value = 0, row=0) -> Value:
        v = b.smem_load_vN(self.smem, self.word(b, name, idx, row), dtype=I32, n=1)
        return b.vec_extract(v, 0)

    def st_i32(
        self, b: IRBuilder, name: str, idx: int | Value, value: Value, row=0
    ) -> None:
        b.smem_store_vN(self.smem, [self.word(b, name, idx, row)], value, 1)

    def ld_f32(self, b: IRBuilder, name: str, idx: int | Value = 0, row=0) -> Value:
        return b.bitcast(self.ld_i32(b, name, idx, row), F32)

    def st_f32(
        self, b: IRBuilder, name: str, idx: int | Value, value: Value, row=0
    ) -> None:
        self.st_i32(b, name, idx, b.bitcast(value, I32), row)

    def ld_vec(
        self, b: IRBuilder, name: str, idx: int | Value, n: int, row=0
    ) -> Value:
        """``<n x i32>`` from ``name[idx .. idx + n)`` (n in 1, 2, 3, 4, 8)."""
        return b.smem_load_vN(self.smem, self.word(b, name, idx, row), dtype=I32, n=n)

    def st_vec(
        self, b: IRBuilder, name: str, idx: int | Value, value: Value, row=0
    ) -> None:
        b.smem_store_vN(
            self.smem, [self.word(b, name, idx, row)], value, value.type.count
        )
