# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""External emit RECORDER + `b`-witness gate for the tiling layout-viz auto-pipeline.

The recorder DECORATES the tiling VERBS for the duration of a build so every mem<->register
movement and register op announces itself as a frozen node -- with **no core edits** and **no
author burden**. It rebinds the verb NAMES in the build function's module globals (where bare
`store_fragment(...)` calls actually resolve) plus the emit / package-alias namespaces, and
`TileMma.__call__` on the class, captures a node BEFORE delegating to the original (so the emitted
IR is byte-identical), and restores every binding on exit -- even on exception.

The `b`-witness completeness gate then counts the memory/mma ops actually emitted into
`kernel.body.ops` and reconciles them against the recorded nodes' fan-out, raising
:class:`CoverageError` on any unaccounted movement -- a LOUD refusal, never a silent wrong render
(a kernel that moves data outside the registered verbs, e.g. a raw ``b.mma``, is caught here).

PROTOTYPE MARKER (Milestone 0 / Phase A): :func:`verify_roundtrip` is a minimal addressing round-trip
that will move to ``visualization/auto_pipeline.py`` in Phase C (with the buffer-half key + the MMA
soundness gate). The recorder + witness below are the durable Phase-A surface.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable, Iterator

# --------------------------------------------------------------------------------------------------
# Recorded node model
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineTransaction:
    """One mem<->register movement (a verb call), captured at the boundary with its descriptors.

    ``fill`` is register-only (no space/addressing); ``load``/``store`` carry the memory descriptors.
    ``origin`` keeps the raw ``window.origin`` entries (ints for constant origins, SSA ``Value``
    handles for symbolic ones) -- resolved later by the driver, never coerced here.
    """

    kind: str  # "load" | "store" | "fill"
    seq: int
    space: str  # "global" | "lds" | "reg" (fill)
    space_id: int  # id() of the ptr/smem Value -- identity key for a memory space
    space_name: str
    tile_desc: Any  # the frozen TileDesc (carries .layout, .register_count, .shape)
    dtype_name: str
    strides: tuple[Any, ...] | None
    lengths: tuple[Any, ...] | None
    origin: tuple[Any, ...] | None
    register_count: int
    vw: int  # per-access fan-out width (registers per hardware op)
    swizzle: Any
    produces: int | None = None       # id() of the SSA Value this node yields (the loaded/filled fragment)
    consumes: tuple[int, ...] = ()     # id()s of the SSA Values this node reads (the stored fragment)

    @property
    def encoding(self) -> Any:
        return self.tile_desc.layout

    @property
    def op_fanout(self) -> int:
        """Hardware mem-ops this verb emits: one wide op per ``vw`` registers."""
        if self.kind == "fill":
            return 0
        return ceil(self.register_count / self.vw)


@dataclass(frozen=True)
class PipelineOp:
    """One register<->register op: a ``transform_fragment`` (reorder|cross_lane) or the MMA."""

    kind: str  # "reorder" | "cross_lane" | "mma"
    seq: int
    src_enc: Any = None
    tgt_enc: Any = None
    a_enc: Any = None       # CONSUMED operand encodings (the TileMma fragments, not a load view)
    b_enc: Any = None
    c_enc: Any = None
    a_canon: Any = None     # the atom's canonical refs (the fixed machine) -- for the soundness gate
    b_canon: Any = None
    c_canon: Any = None
    atom_shape: tuple[int, ...] | None = None
    atom_count: int = 0  # number of `tile.mma` the TileMma call explodes into
    note: str = ""
    produces: int | None = None       # id() of the SSA Value this op yields (transform / mma output)
    consumes: tuple[int, ...] = ()     # id()s of the SSA Values this op reads (operands / transform src)


@dataclass
class RecordedPipeline:
    """The ordered chain of transactions + ops captured from one build, plus the memory spaces."""

    nodes: list[Any] = field(default_factory=list)
    spaces: dict[int, str] = field(default_factory=dict)  # space_id -> space_name
    arch: str | None = None       # gfx target (from the TileMma) -- the DRIVER's SoT for nbanks/wave_size
    wave_size: int | None = None  # lanes per wave (from the TileMma traits) -- NOT assumed 64

    @property
    def transactions(self) -> list[PipelineTransaction]:
        return [n for n in self.nodes if isinstance(n, PipelineTransaction)]

    @property
    def ops(self) -> list[PipelineOp]:
        return [n for n in self.nodes if isinstance(n, PipelineOp)]

    def lds_spaces(self) -> list[int]:
        """space_ids that are written AND read in LDS (round-trip candidates), in first-seen order."""
        seen: list[int] = []
        for t in self.transactions:
            if t.space == "lds" and t.space_id not in seen:
                seen.append(t.space_id)
        return seen

    def block_diagram(self, out_path: str, *, title: str = "") -> str:
        """Render the Level-0 pipeline block diagram (labelled boxes + flow) to ``out_path``. Thin
        wrapper over :func:`visualization.block_diagram.block_diagram` -- matplotlib stays lazy."""
        from .visualization.block_diagram import block_diagram
        return block_diagram(self, out_path, title=title)


class CoverageError(RuntimeError):
    """Raised when emitted memory/mma ops are NOT fully accounted by the recorded nodes."""


# --------------------------------------------------------------------------------------------------
# The recorder: decorate the verbs (external), capture-before-delegate, restore on exit
# --------------------------------------------------------------------------------------------------

# The documented coverage surface: a new tiling verb opts in by joining this registry.
_MEMORY_VERBS = ("load_fragment", "store_fragment", "fill_fragment")
_OP_VERBS = ("transform_fragment",)


class _Recorder:
    """Mutable accumulator the verb wrappers append to (closed over by the wrappers)."""

    def __init__(self) -> None:
        self.pipeline = RecordedPipeline()
        self._seq = 0

    def _next(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def add_memory(self, kind: str, ptr: Any, window: Any, tile_desc: Any, swizzle: Any,
                   *, produces: int | None = None, consumes: tuple[int, ...] = ()) -> None:
        from .emit import _is_lds

        space = "lds" if _is_lds(ptr) else "global"
        tensor = window.tensor
        # Mirror emit's vw EXACTLY: a global store is scalar (store_fragment `vw = ... if lds else 1`,
        # emit.py:326); everything else takes the contiguous run (+ swizzle cap for LDS).
        if kind == "store" and space == "global":
            vw = 1
        else:
            vw = _access_vw(tile_desc, tensor.strides, tensor.dtype.name, swizzle)
        node = PipelineTransaction(
            kind=kind, seq=self._next(), space=space, space_id=id(ptr),
            space_name=getattr(ptr, "name", "?"), tile_desc=tile_desc,
            dtype_name=tensor.dtype.name, strides=tuple(tensor.strides),
            lengths=tuple(tensor.lengths), origin=tuple(window.origin),
            register_count=tile_desc.register_count, vw=vw, swizzle=swizzle,
            produces=produces, consumes=tuple(consumes),
        )
        self.pipeline.nodes.append(node)
        self.pipeline.spaces[node.space_id] = node.space_name

    def add_fill(self, fragment: Any) -> None:
        td = fragment.tile_desc
        self.pipeline.nodes.append(PipelineTransaction(
            kind="fill", seq=self._next(), space="reg", space_id=id(fragment),
            space_name="reg", tile_desc=td, dtype_name=getattr(fragment.dtype, "name", "?"),
            strides=None, lengths=None, origin=None,
            register_count=td.register_count, vw=1, swizzle=False,
            produces=id(fragment.value), consumes=(),
        ))

    def add_transform(self, fragment: Any, target_desc: Any, *, produces: int | None = None,
                      consumes: tuple[int, ...] = ()) -> None:
        from .transforms import classify_transform

        src = fragment.tile_desc.layout
        tgt = target_desc.layout
        kind = classify_transform(src, tgt)
        self.pipeline.nodes.append(PipelineOp(
            kind=kind if kind in ("reorder", "cross_lane") else "reorder",
            seq=self._next(), src_enc=src, tgt_enc=tgt, note=str(kind),
            produces=produces, consumes=tuple(consumes),
        ))

    def add_mma(self, mma: Any, a_fragment: Any, b_fragment: Any, accumulator: Any, *,
                produces: int | None = None, consumes: tuple[int, ...] = ()) -> None:
        atom_count = mma._m_subtiles * mma._n_subtiles * mma._k_subtiles
        # a_enc/b_enc = the CONSUMED operand encodings (the fragments the kernel feeds in -- for CRC these
        # are its OWN interleaved distributions, NOT TileMma's broken interleaved output). a_canon/b_canon =
        # the CANONICAL machine refs (mma.a_layout, interleaved=False -- the trusted path the tee uses).
        self.pipeline.nodes.append(PipelineOp(
            kind="mma", seq=self._next(),
            a_enc=a_fragment.tile_desc.layout, b_enc=b_fragment.tile_desc.layout,
            c_enc=accumulator.tile_desc.layout,
            a_canon=mma.a_layout, b_canon=mma.b_layout, c_canon=mma.c_layout,
            atom_shape=tuple(mma._atom_shape), atom_count=atom_count, note="TileMma",
            produces=produces, consumes=tuple(consumes),
        ))
        if self.pipeline.arch is None:                         # the recording's SoT for arch/wave-size
            self.pipeline.arch = mma.target
            self.pipeline.wave_size = mma.wave_size


def _access_vw(tile_desc: Any, strides: Any, dtype_name: str, swizzle: Any) -> int:
    """The vw the emit would choose for this access (reuse the validated bank-conflict helper)."""
    from .lds_conflict import access_width

    return access_width(tile_desc, tuple(strides), dtype_name, swizzle)


def _namespaces(build_fn: Callable) -> list[dict]:
    """Every namespace where a bare verb name could resolve: the build fn's module globals + the
    emit / transforms / package-`__init__` modules (for qualified callers)."""
    import rocke.helpers.tiling as _pkg
    from . import emit as _emit
    from . import transforms as _transforms

    return [build_fn.__globals__, _emit.__dict__, _transforms.__dict__, _pkg.__dict__]


@contextmanager
def record_pipeline(build_fn: Callable) -> Iterator[RecordedPipeline]:
    """Decorate the tiling verbs for the duration of the block, yield the accumulating pipeline,
    and restore every binding on exit (even on exception)."""
    from . import emit as _emit
    from . import transforms as _transforms
    from .mma.mma_operation import TileMma

    rec = _Recorder()
    namespaces = _namespaces(build_fn)
    originals = {
        "load_fragment": _emit.load_fragment,
        "store_fragment": _emit.store_fragment,
        "fill_fragment": _emit.fill_fragment,
        "transform_fragment": _transforms.transform_fragment,
    }

    def _wrap_load(orig):
        def w(b, ptr, window, tile_desc, thread, **kw):
            res = orig(b, ptr, window, tile_desc, thread, **kw)   # delegate first: capture the produced value
            rec.add_memory("load", ptr, window, tile_desc, kw.get("lds_swizzle", False),
                           produces=id(res.value))
            return res
        return w

    def _wrap_store(orig):
        def w(b, ptr, window, fragment, thread, **kw):
            rec.add_memory("store", ptr, window, fragment.tile_desc, kw.get("lds_swizzle", False),
                           consumes=(id(fragment.value),))
            return orig(b, ptr, window, fragment, thread, **kw)
        return w

    def _wrap_fill(orig):
        def w(b, fragment, scalar):
            res = orig(b, fragment, scalar)                       # fragment.value is bound by now
            rec.add_fill(fragment)
            return res
        return w

    def _wrap_transform(orig):
        def w(b, fragment, target_desc, **kw):
            res = orig(b, fragment, target_desc, **kw)
            rec.add_transform(fragment, target_desc, produces=id(res.value),
                              consumes=(id(fragment.value),))
            return res
        return w

    wrappers = {
        "load_fragment": _wrap_load(originals["load_fragment"]),
        "store_fragment": _wrap_store(originals["store_fragment"]),
        "fill_fragment": _wrap_fill(originals["fill_fragment"]),
        "transform_fragment": _wrap_transform(originals["transform_fragment"]),
    }

    # Save + rebind every namespace slot that currently holds the original verb.
    restore: list[tuple[dict, str, Any]] = []
    for name, orig in originals.items():
        for ns in namespaces:
            if ns.get(name, None) is orig:
                restore.append((ns, name, orig))
                ns[name] = wrappers[name]

    mma_orig = TileMma.__call__

    def _wrap_mma_call(self, b, a_fragment, b_fragment, accumulator):
        out = mma_orig(self, b, a_fragment, b_fragment, accumulator)
        rec.add_mma(self, a_fragment, b_fragment, accumulator, produces=id(out.value),
                    consumes=(id(a_fragment.value), id(b_fragment.value), id(accumulator.value)))
        return out

    TileMma.__call__ = _wrap_mma_call
    try:
        yield rec.pipeline
    finally:
        for ns, name, orig in restore:
            ns[name] = orig
        TileMma.__call__ = mma_orig


def record_build(build_fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, RecordedPipeline]:
    """Run ``build_fn(*args, **kwargs)`` with the verbs decorated; return ``(result, pipeline)``."""
    with record_pipeline(build_fn) as pipeline:
        result = build_fn(*args, **kwargs)
    return result, pipeline


# --------------------------------------------------------------------------------------------------
# The `b`-witness completeness gate
# --------------------------------------------------------------------------------------------------

_MEM_OP_NAMES = frozenset({
    "memref.global_load_vN", "memref.global_load", "memref.masked_global_load",
    "memref.global_store_typed", "memref.global_store_vN",
    "tile.smem_load_vN", "tile.smem_store_vN",
})
_MMA_OP_NAMES = frozenset({"tile.mma"})


def op_histogram(kernel: Any) -> dict[str, int]:
    """Count every op name in the kernel body, recursing into nested regions (scf.for/scf.if)."""
    hist: dict[str, int] = {}

    def walk(region: Any) -> None:
        for op in region.ops:
            hist[op.name] = hist.get(op.name, 0) + 1
            for sub in op.regions:
                walk(sub)

    walk(kernel.body)
    return hist


@dataclass
class WitnessReport:
    mem_expected: int
    mem_counted: int
    mma_expected: int
    mma_counted: int
    histogram: dict[str, int]

    @property
    def mem_ok(self) -> bool:
        return self.mem_expected == self.mem_counted

    @property
    def mma_ok(self) -> bool:
        return self.mma_expected == self.mma_counted

    @property
    def ok(self) -> bool:
        return self.mem_ok and self.mma_ok


def witness(pipeline: RecordedPipeline, kernel: Any, *, raise_on_gap: bool = True) -> WitnessReport:
    """Reconcile emitted memory/mma ops against the recorded nodes' fan-out.

    Raises :class:`CoverageError` (when ``raise_on_gap``) if either category is unaccounted -- the
    LOUD refusal that turns a cross-module / non-verb miss (e.g. a raw ``b.mma``) into a hard stop
    rather than a silently short pipeline.
    """
    hist = op_histogram(kernel)
    mem_expected = sum(t.op_fanout for t in pipeline.transactions if t.kind in ("load", "store"))
    mem_counted = sum(hist.get(n, 0) for n in _MEM_OP_NAMES)
    mma_expected = sum(o.atom_count for o in pipeline.ops if o.kind == "mma")
    mma_counted = sum(hist.get(n, 0) for n in _MMA_OP_NAMES)
    report = WitnessReport(mem_expected, mem_counted, mma_expected, mma_counted, hist)
    if raise_on_gap and not report.ok:
        raise CoverageError(
            "unrecorded data movement -- viz can't cover this kernel: "
            f"mem expected={mem_expected} counted={mem_counted}, "
            f"mma expected={mma_expected} counted={mma_counted} "
            f"(histogram={hist})"
        )
    return report


# --------------------------------------------------------------------------------------------------
# Minimal addressing round-trip (PROTOTYPE -> moves to auto_pipeline.py in Phase C)
# --------------------------------------------------------------------------------------------------


def _elem_addresses(t: PipelineTransaction, n_lanes: int = 64) -> dict[tuple[int, int], int]:
    """(lane, register) -> element address for an LDS transaction, replaying the real emit map."""
    from .lds_conflict import addr_map

    origin = tuple(int(o) for o in t.origin)  # Milestone 0: constant LDS origins
    accesses, vw = addr_map(t.tile_desc, tuple(t.strides), origin=origin, n_lanes=n_lanes,
                            dtype_name=t.dtype_name, lds_swizzle=t.swizzle)
    out: dict[tuple[int, int], int] = {}
    for a in accesses:
        for i in range(a["vw"]):
            out[(a["lane"], a["reg0"] + i)] = a["base"] + i
    return out


def _labels(t: PipelineTransaction, n_lanes: int = 64) -> dict[tuple[int, int], tuple[int, ...]]:
    """(lane, register) -> the logical tile coordinate the encoding places there."""
    from .register_mapper import RegisterMapper

    rm = RegisterMapper(t.encoding)
    return {
        (lane, reg): rm.matrix_coordinates(lane, reg)
        for lane in range(n_lanes)
        for reg in range(rm.num_vector_items)
    }


def verify_roundtrip(pipeline: RecordedPipeline, *, n_lanes: int = 64) -> list[int]:
    """For each LDS space with a store then a read, assert the read recovers, at each address, the
    logical label the store placed there. Returns the space_ids verified. Raises AssertionError with
    the first offending ``(lane, reg)`` on failure.

    Milestone 0 pairs the FIRST store with the FIRST read per space (the toy is single-buffer, so
    every K-iteration's pair is identical). The buffer-half key + MMA-soundness gate arrive in the
    Phase-C driver.
    """
    verified: list[int] = []
    for space_id in pipeline.lds_spaces():
        stores = [t for t in pipeline.transactions if t.space_id == space_id and t.kind == "store"]
        reads = [t for t in pipeline.transactions if t.space_id == space_id and t.kind == "load"]
        if not stores or not reads:
            continue
        store, read = stores[0], reads[0]
        store_addr = _elem_addresses(store, n_lanes)
        store_label = _labels(store, n_lanes)
        content: dict[int, tuple[int, ...]] = {}
        for key, addr in store_addr.items():
            lbl = store_label[key]
            if addr in content and content[addr] != lbl:
                raise AssertionError(
                    f"store collision on space {store.space_name!r} addr {addr}: "
                    f"{content[addr]} vs {lbl}"
                )
            content[addr] = lbl
        read_addr = _elem_addresses(read, n_lanes)
        read_label = _labels(read, n_lanes)
        for key, addr in read_addr.items():
            if addr not in content:
                raise AssertionError(
                    f"round-trip GAP on space {read.space_name!r}: read {key} -> addr {addr} "
                    f"was never written"
                )
            got, expected = content[addr], read_label[key]
            if got != expected:
                raise AssertionError(
                    f"round-trip MISMATCH on space {read.space_name!r}: read {key} -> addr {addr} "
                    f"got label {got}, expected {expected}"
                )
        verified.append(space_id)
    return verified
