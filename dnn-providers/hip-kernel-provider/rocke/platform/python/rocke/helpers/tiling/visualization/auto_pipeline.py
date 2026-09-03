# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Phase-C driver for the automated layout-viz pipeline.

Turns a :class:`~rocke.helpers.tiling.tiling_recorder.RecordedPipeline` into verified, renderable
stages. Landing incrementally with the CRC MVP; the first piece is the **symbolic-origin resolver**.

A captured ``window.origin`` entry is an SSA ``Value`` whose ``.op`` back-refs form a walkable DAG that
bottoms out at named ROOTS -- the ``scf.for`` induction variable (the K-loop iteration), ``gpu.thread_id``
(the wave/lane), ``gpu.block_id`` (the macro-tile block), and ``arith.constant`` leaves -- combined by
``arith.add/sub/mul/div/mod``. Resolving an origin = pinning those roots (``k`` / ``tid`` / ``block_id``)
and evaluating the DAG to a concrete int, so a symbolic double-buffered / cooperative origin becomes the
buffer-half + wave offset for a chosen slice.
"""
from __future__ import annotations

from typing import Any

from . import block_diagram as _bd

_BINARY = {
    "arith.add": lambda a, b: a + b,
    "arith.sub": lambda a, b: a - b,
    "arith.mul": lambda a, b: a * b,
    "arith.div": lambda a, b: a // b,   # emit uses integer division
    "arith.mod": lambda a, b: a % b,
}


class OriginResolutionError(RuntimeError):
    """Raised when an origin DAG hits an op/root the resolver was not told how to pin."""


def resolve_value(value: Any, bindings: dict) -> int:
    """Resolve an SSA ``Value`` (or a plain int) to a concrete int by walking ``Value.op`` to the
    pinned roots.

    ``bindings`` supplies the substitution roots:
      - ``k``:        the ``scf.for`` induction value (e.g. the K-tile offset ``kb * tile_k``),
      - ``tid``:      the thread id (the wave = ``tid // wave_size``, lane = ``tid % wave_size``),
      - ``block_id``: optional ``{axis: int}`` for the macro-tile block (global origins).

    The loop IV is a substitution ROOT, not a constant leaf -- any value produced by the ``scf.for``
    op resolves to ``bindings['k']``.
    """
    if isinstance(value, int):
        return value
    op = getattr(value, "op", None)
    if op is None:
        raise OriginResolutionError(
            f"value {getattr(value, 'name', value)!r} has no producing op and is not an int"
        )
    name = op.name
    if name == "arith.constant":
        return int(op.attrs["value"])
    if name == "scf.for":
        if "k" not in bindings:
            raise OriginResolutionError("scf.for induction variable hit but no 'k' binding supplied")
        return int(bindings["k"])
    if name == "gpu.thread_id":
        if "tid" not in bindings:
            raise OriginResolutionError("gpu.thread_id hit but no 'tid' binding supplied")
        return int(bindings["tid"])
    if name == "gpu.block_id":
        axis = op.attrs.get("axis")
        block = bindings.get("block_id", {})
        if axis not in block:
            raise OriginResolutionError(f"gpu.block_id[{axis}] hit but no 'block_id' binding supplied")
        return int(block[axis])
    fn = _BINARY.get(name)
    if fn is None:
        raise OriginResolutionError(f"unhandled op in origin DAG: {name!r}")
    return fn(resolve_value(op.operands[0], bindings), resolve_value(op.operands[1], bindings))


def resolve_origin(origin: tuple, bindings: dict) -> tuple[int, ...]:
    """Resolve every axis of a captured ``window.origin`` to a concrete int tuple."""
    return tuple(resolve_value(o, bindings) for o in origin)


# --------------------------------------------------------------------------------------------------
# Addressing round-trip (gate 1) -- per LDS buffer-half, keyed by smem identity + resolved half
# --------------------------------------------------------------------------------------------------


class RoundTripError(RuntimeError):
    """Raised when a read touches an LDS address the cooperative store never wrote (per half)."""


def _lane_span(encoding: Any) -> int:
    """Total threads an encoding spans = product over ALL lane-partition levels (wave outer + lane
    inner). `RegisterMapper.num_lanes` reads only the first level, so for a cooperative NDimP=2
    encoding it undercounts -- this is the count `emit_tensor_coordinates` actually decomposes."""
    span = 1
    for majors, minors in zip(encoding.lane_to_rh_major, encoding.lane_to_rh_minor):
        for major, minor in zip(majors, minors):
            span *= encoding.bucket_length(major, minor)
    return span


def verify_lds_roundtrip(pipeline: Any, space_id: int, *, tile_k: int) -> list[int]:
    """Addressing round-trip for one double-buffered LDS space, PER buffer-half.

    Store and read descriptors are both ``_transpose_desc``'d into the same ``(K, free)`` frame, so the
    element ADDRESS and the ``(K, free)`` label coincide -- the gate reduces to: every read address (over
    all waves, at half ``h``) was written by the cooperative store to half ``h``, and the store is a
    bijection with no leak across the half boundary. Origins are resolved via :func:`resolve_origin`;
    the half is keyed on the RESOLVED free-origin, never the raw symbolic ``oth`` expression (the store
    writes ``oth`` while the same-iteration read reads ``cur`` -- pairing those would be a silent bug).

    ``tile_k`` pins the K-loop iteration for the resolver (from the render context). Returns the list of
    verified halves; raises :class:`RoundTripError` naming the first offending read.
    """
    from ..lds_conflict import addr_map

    _arch, wave_size = _arch_wave(pipeline)                    # DERIVED from the recording, not defaulted
    txns = [t for t in pipeline.transactions if t.space_id == space_id]
    stores = [t for t in txns if t.kind == "store"]
    reads = [t for t in txns if t.kind == "load"]
    if not stores or not reads:
        return []

    store_desc, read_desc = stores[0].tile_desc, reads[0].tile_desc
    strides = tuple(stores[0].strides)
    free_stride = strides[0]  # K-row stride = bufs*tile_m; the free axis (stride 1) spans both buffers
    store_lanes = _lane_span(store_desc.layout)  # cooperative: all waves' threads (e.g. 256)
    n_waves = store_lanes // wave_size
    read_dag = reads[0].origin

    # Halves = the distinct resolved store free-origins across the K-tiles (prologue writes half 0; the
    # in-loop store writes `oth` -> the other half). tile_m = the per-half free extent.
    store_frees = {resolve_origin(t.origin, {"k": kb * tile_k, "tid": 0})[1]
                   for t in stores for kb in range(2)}
    bufs = len(store_frees)
    tile_m = free_stride // bufs

    name = pipeline.spaces[space_id]

    def elem_addrs(desc, origin, n_lanes, dtype, swizzle):
        acc, _ = addr_map(desc, strides, origin=origin, n_lanes=n_lanes,
                          dtype_name=dtype, lds_swizzle=swizzle)
        for a in acc:
            for i in range(a["vw"]):
                yield a["base"] + i

    verified: list[int] = []
    for h in range(bufs):
        lo, hi = h * tile_m, (h + 1) * tile_m
        # (a) confinement -- the store's DATA (pre-swizzle) stays within half h; a swizzle then permutes
        # WITHIN the buffer, so this leak check is a pre-swizzle property (post-swizzle freely crosses lo/hi).
        for addr in elem_addrs(store_desc, (0, h * tile_m), store_lanes, stores[0].dtype_name, False):
            free = addr % free_stride
            if not (lo <= free < hi):
                raise RoundTripError(f"{name}: store data leaks half {h} at free {free}")
        # (b) coverage -- every read address (all waves, at the REAL swizzle) was written by the coop store.
        written: set[int] = set()
        for addr in elem_addrs(store_desc, (0, h * tile_m), store_lanes, stores[0].dtype_name,
                               stores[0].swizzle):
            if addr in written:
                raise RoundTripError(f"{name}: store collision half {h} addr {addr}")
            written.add(addr)
        for w in range(n_waves):
            r_origin = resolve_origin(read_dag, {"k": h * tile_k, "tid": w * wave_size})
            if r_origin[1] // tile_m != h:
                raise RoundTripError(
                    f"{name}: read wave {w} resolves to half {r_origin[1] // tile_m}, expected {h}")
            for addr in elem_addrs(read_desc, r_origin, wave_size, reads[0].dtype_name, reads[0].swizzle):
                if addr not in written:
                    raise RoundTripError(
                        f"{name} half {h} wave {w}: read -> addr {addr} "
                        f"(K={addr // free_stride}, free={addr % free_stride}) never written")
        verified.append(h)
    return verified


# --------------------------------------------------------------------------------------------------
# MMA soundness (gate 2) -- orthogonal to the addressing round-trip
# --------------------------------------------------------------------------------------------------


class MmaSoundnessError(RuntimeError):
    """Raised when a recorded MMA's consumed operands are not a sound, K-aligned pair (the sound MAC)."""


def verify_mma_soundness(pipeline: Any) -> int:
    """Gate 2: every recorded MMA's CONSUMED operand encodings must be sound MMA operands against the
    canonical machine (``operand_soundness``) AND share a K-distribution (``diagnose_k_match``) -- bundled
    as ``mma_pair_compatible``. This is the operand-correctness the addressing round-trip is BLIND to: a
    scrambled/duplicated operand K passes the address gate but is caught here. ``a_canon``/``b_canon`` are
    the trusted canonical layouts from the MMA definition (``interleaved=False``); ``a_enc``/``b_enc`` are
    the kernel's own (interleaved) operands. Returns the count verified; raises on the first unsound MMA.
    Correctness SOT: ``docs/mma_is_machinery.md`` (the three-condition sound MAC).
    """
    from ..transforms import mma_pair_compatible

    verified = 0
    for op in pipeline.ops:
        if op.kind != "mma":
            continue
        d = mma_pair_compatible(op.a_enc, op.b_enc, a_canon=op.a_canon, b_canon=op.b_canon)
        if d.severity == "error":
            raise MmaSoundnessError(f"MMA op seq {op.seq}: {d.message}")
        verified += 1
    return verified


# --------------------------------------------------------------------------------------------------
# Render driver (Phase C render half) -- map recorded transactions onto the existing FlowStage recipes
# --------------------------------------------------------------------------------------------------

_ARCH_NBANKS = {"gfx90a": 32, "gfx942": 32}


def _elem_bytes(dtype_name: str) -> int:
    from ..emit import _BYTE_WIDTH

    return _BYTE_WIDTH[dtype_name]


def _arch_wave(pipeline: Any) -> tuple[str, int]:
    """The (arch, wave_size) the driver DERIVES from the recording (captured from the TileMma). Fails loud
    if absent -- NO silent 'gfx90a'/64 fallback; a kernel without a recorded TileMma must supply them via
    a different driver."""
    if pipeline.arch is None or pipeline.wave_size is None:
        raise ValueError("pipeline has no recorded arch/wave_size (no TileMma was recorded) -- cannot "
                         "derive them; record a kernel with a TileMma, or drive the utilities directly")
    return pipeline.arch, pipeline.wave_size


def _lds_geometry(pipeline: Any, space_id: int, *, tile_k: int):
    """Derive the physical LDS render params for a space from its recorded transactions: the coop store /
    wave-read descriptors, memref stride, n_waves, per-half free extent (tile_m), elem_bytes. The wave size
    is DERIVED from the recording (never assumed)."""
    _arch, wave_size = _arch_wave(pipeline)
    txns = [t for t in pipeline.transactions if t.space_id == space_id]
    store = next(t for t in txns if t.kind == "store")
    read = next(t for t in txns if t.kind == "load")
    strides = tuple(store.strides)
    n_waves = _lane_span(store.tile_desc.layout) // wave_size
    store_frees = {resolve_origin(t.origin, {"k": kb * tile_k, "tid": 0})[1]
                   for t in txns if t.kind == "store" for kb in range(2)}
    tile_m = strides[0] // max(1, len(store_frees))
    return store, read, strides, n_waves, tile_m, _elem_bytes(store.dtype_name)


def render_lds_store(pipeline: Any, space_id: int, *, tile_k: int, dims: tuple[str, str] = ("M", "K"),
                     wave: int = 0, buffer: int = 0, out_path: str) -> str:
    """Render the cooperative LDS-STORE stage (wave registers -> LDS) of an LDS space, built directly per
    the MMA-expert-reconciled conventions and GATED in code (Cardinal Rule 12) before rendering:

    - cell LABEL = the datum's INVARIANT logical `(free, K)` identity — `dims=("M","K")` for A, `("N","K")`
      for B. The store descriptor's OWN X-dims are `(K, free)` (it indexes the `(K, free)` LDS memref); the
      coord-swap just presents that datum in logical-axis order, it does NOT transpose the datum or invent a
      label. Register file and LDS carry the SAME invariant label.
    - the store is a REPOSITION into LDS (`describe_edge`, `to_space="lds"`): the label is invariant; only the
      position changes (register slot -> LDS address). On CRC there is **no transpose** — the free-contiguous
      coop load lands on the free-stride-1 LDS banks (the CRC free-major gift), symmetric for A and B.
    - SHADE = `register_count / vw` transactions, in reg-issue order (`reg // vw`) — vw honours the swizzle's
      width cap (`access_width`), NOT `transactions()` (which can't see a width-capped contiguous run).
    - `dtype_bits = elem_bytes*8` (f16 -> 4 physical 32-bit registers, not 8); `color_mode="first8"`.
    """
    from .kernel_stages import coop_forward_map, lds_inputs
    from .layout_render import FlowStage, LdsBankView, Pipeline, RegisterFileComponent
    from ..transforms import describe_edge

    arch, wave_size = _arch_wave(pipeline)                     # DERIVED from the recording, not defaulted
    store, read, strides, n_waves, tile_m, eb = _lds_geometry(pipeline, space_id, tile_k=tile_k)
    # Everything below is derived from the dtype WIDTH and the recorded descriptors -- NO magic numbers.
    dtype_bits = eb * 8                          # element width in bits (f16->16, f32->32)
    vw = store.vw                                # emitted vector width, in ELEMENTS (recorded from emit)
    elems_per_reg = max(1, 32 // dtype_bits)     # elements per 32-bit register (f16->2, f32->1)
    b128_elems = max(1, 128 // dtype_bits)       # a 128-bit (b128/dwordx4) access in ELEMENTS (f16->8, f32->4)
    phys_regs = -(-store.register_count // elems_per_reg)   # physical 32-bit registers (ceil)

    lo = wave * wave_size
    full = coop_forward_map(store.tile_desc, n_waves=n_waves, wave_size=wave_size)
    pos = {(t - lo, r): c for (t, r), c in full.items() if lo <= t < lo + wave_size}   # (K, free) memref coord
    # LABEL = the datum's INVARIANT logical (free, K) identity, presented in logical-axis order by swapping the
    # store descriptor's own (K, free) memref frame -- NOT a transpose of the datum, NOT a manufactured label.
    # The store REPOSITIONS the datum (register slot -> LDS address); the label is invariant, and on CRC there
    # is no transpose (free-contiguous -> free-stride-1 banks). Register file and LDS carry this same label.
    label = {k: (c[1], c[0]) for k, c in pos.items()}                                  # logical (free, K) datum
    _, addr_fn = lds_inputs(store.tile_desc, stride=strides[0], swizzle=store.swizzle or None)
    # SHADE = §9.6 transaction time order: ONE shade per vectorized LDS write, width from the RECORDED store
    # addressing (strides + swizzle, baked into `addr_fn`), CAPPED at b128 (`vector_transactions`) -- ✗ NEVER the
    # emit `vw` (pre-coalescing) or the register index. A run wider than b128 splits into multiple shades.
    from .layout_render import vector_transactions
    shade, n_shades = vector_transactions(pos, addr_fn, dtype_bits)
    edge_kind, edge_why = describe_edge(label, None, src_dims=dims, to_space="lds")     # reposition, label invariant

    # --- Cardinal Rule 12: verify IN CODE before rendering (all dtype-derived; NO magic numbers) ---
    assert 1 <= vw <= b128_elems, f"emitted vw {vw} > the 128-bit max of {b128_elems} {store.dtype_name} elems"
    assert n_shades >= 1 and max(shade.values()) + 1 == n_shades, "shade must be a dense 0..n_shades-1 transaction order"
    assert all(len({shade[(l, r)] for r in range(store.register_count)}) == n_shades
               for l in range(wave_size)), "shade (per-thread b128 transactions) must be lockstep across threads"
    assert phys_regs * elems_per_reg == store.register_count, "register_count must pack evenly into 32-bit regs"
    assert store.tile_desc.layout is not read.tile_desc.layout, "this is the store hop, not the read"
    assert len(pos) == wave_size * store.register_count and max(l for l, _ in pos) == wave_size - 1
    # The store is a REPOSITION into LDS (register slot -> LDS address), label INVARIANT. On CRC it is a plain
    # free-contiguous store onto free-stride-1 banks -- NO transpose (symmetric for A and B). We assert the
    # classification and the label invariance (below); we do NOT assert a transpose, because there isn't one.
    assert edge_kind == "reposition", f"LDS store must classify as a reposition, got {edge_kind}"

    reg = RegisterFileComponent(fwd_map=label, dims=dims, dtype_bits=dtype_bits,
                                color_mode="first8", shade_map=shade)
    lds = LdsBankView(mp=pos, addr_fn=addr_fn, flow_map=label, nbanks=_ARCH_NBANKS[arch], elem_bytes=eb,
                      lds_base_bytes=buffer * tile_m * eb, dims=dims)
    # TRACEABILITY: register file and LDS carry the SAME invariant label for every datum (one identity, two
    # positions -- the register slot and the LDS bank).
    assert reg.fwd_map == lds.flow_map, "register-file and LDS labels diverged -- label is not invariant"
    pipe = Pipeline(stages=(FlowStage(f"wave {wave} registers (tid x vreg)", reg, source="store.tile_desc",
                                      dist=store.tile_desc.layout),
                            FlowStage(f"LDS (bank = dword mod {_ARCH_NBANKS[arch]})", lds,
                                      source="store.tile_desc",
                                      transform=f"LDS store — {edge_kind}\n[{edge_why}]",
                                      dist=store.tile_desc.layout)),
                    title=f"{pipeline.spaces[space_id]} LDS store (wave {wave}, buf {buffer})")
    return pipe.render_panels(out_path)


# --------------------------------------------------------------------------------------------------
# Generic flow SEGMENTATION -- any kernel's recording -> the L1 flows to draw, each transition ONCE
# --------------------------------------------------------------------------------------------------
#
# The recording is a dataflow graph of TRANSITIONS (transactions + ops) between memory/register states.
# A FLOW is a maximal producer->consumer chain between memory endpoints; every recorded transition
# belongs to EXACTLY ONE flow, so "each transition drawn once" is emergent (not special-cased). Detection
# reuses the block-diagram's own generic segmentation (`extract_blocks` + `_dataflow_edges` +
# `COMBINING_OPS`) -- NOTHING here keys on GEMM nouns (A/B/C are lane TAGS for naming only, from the
# recorded space name). Roles are algorithm-agnostic shapes:
#   - prefetch : a global-load -> LDS-store staging chain whose LDS is later read INTO a combining op.
#   - copy     : a global<->LDS (or LDS->global) staging chain that does NOT feed a combining op.
#   - lds_read : an LDS read that feeds a combining op (the operand readback), its own flow.
#   - load     : a global load that feeds a combining op DIRECTLY (register prefetch, no LDS).
#   - compute  : the combining op(s) + the accumulator fill (rendered as the tee).
#   - epilogue : a combining-op output -> {reorder/cross_lane} -> global store.
# A kernel with no combining op simply yields copy/standalone flows and NO compute (correct).


from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class Flow:
    """One L1 flow: a group of recorded transitions drawn together, exactly once. ``role`` is the generic
    shape (above); ``lane`` is the recorded operand tag (A/B/C or ``*``) used only for naming; ``seqs`` are
    the covered recorded-node seqs (the dedup unit -- their union over all flows == the whole recording);
    ``repr_seq`` is the node the renderer keys off (e.g. the loop-phase instance of a double-buffered leg)."""
    role: str
    lane: str
    seqs: tuple[int, ...]
    repr_seq: int


def _forward_reach(edges: list[tuple[int, int]]):
    """Return ``reach(seq) -> set`` of nodes reachable FORWARD (producer->consumer) from ``seq``."""
    succ: dict[int, list[int]] = {}
    for f, t in edges:
        succ.setdefault(f, []).append(t)

    def reach(seq: int) -> set[int]:
        seen: set[int] = set()
        stack = [seq]
        while stack:
            for m in succ.get(stack.pop(), []):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return seen

    return reach


def _flow_role(b, *, reach, combine, gstore, lanes_feeding_compute) -> str:
    """Classify one block into a generic flow role from its (space, kind) + graph reachability -- no GEMM
    assumption. ``lanes_feeding_compute`` = the operand lanes whose LDS is read into a combining op."""
    sp, k = b.space, b.kind
    if k in _bd.COMBINING_OPS or k == "fill":
        return "compute"
    if k in ("reorder", "cross_lane"):
        return "epilogue"
    hits_combine = bool(reach(b.seq) & combine)
    hits_gstore = bool(reach(b.seq) & gstore)
    if sp == "global" and k == "store":
        return "epilogue" if combine else "copy"       # a store fed by a combine is the epilogue sink
    if sp == "lds" and k == "load":
        return "lds_read" if hits_combine else "copy"
    if sp == "lds" and k == "store":
        return "prefetch" if b.lane in lanes_feeding_compute else "copy"
    if sp == "global" and k == "load":
        if b.lane in lanes_feeding_compute or (hits_combine and not hits_gstore and _reaches_lds_store(b, reach)):
            return "prefetch"
        if hits_combine:
            return "load"                              # direct global->register operand load (no LDS)
        return "copy"
    return "standalone"


def _reaches_lds_store(b, reach) -> bool:
    # placeholder kept simple: a global load in a staging kernel reaches its LDS store via one edge; the
    # lane-feeds-compute test already covers the GEMM case, so this only guards a rare direct-detect path.
    return True


def segment_flows(pipeline: Any) -> list[Flow]:
    """Segment ANY recorded pipeline into its L1 flows (each recorded transition in exactly one). Reuses the
    block-diagram segmentation; groups nodes by (role, lane); asserts full, disjoint coverage."""
    blocks, lo, hi = _bd.extract_blocks(pipeline)
    edges = _bd._dataflow_edges(blocks)
    reach = _forward_reach(edges)
    combine = {b.seq for b in blocks if b.kind in _bd.COMBINING_OPS}
    gstore = {b.seq for b in blocks if b.space == "global" and b.kind == "store"}
    # lanes whose LDS is READ into a combining op -> their staging is a prefetch (else a plain copy).
    lanes_feeding_compute = {b.lane for b in blocks
                             if b.space == "lds" and b.kind == "load" and (reach(b.seq) & combine)}
    groups: dict[tuple[str, str], list[int]] = {}
    for b in blocks:
        role = _flow_role(b, reach=reach, combine=combine, gstore=gstore,
                          lanes_feeding_compute=lanes_feeding_compute)
        lane = b.lane if role not in ("compute", "epilogue") else "C"
        groups.setdefault((role, lane), []).append(b.seq)
    in_loop = set(range(lo, hi + 1)) if lo is not None else set()
    flows: list[Flow] = []
    for (role, lane), seqs in groups.items():
        seqs = tuple(sorted(seqs))
        repr_seq = next((s for s in seqs if s in in_loop), seqs[0])   # prefer the loop-phase instance
        flows.append(Flow(role=role, lane=lane, seqs=seqs, repr_seq=repr_seq))
    covered = [s for f in flows for s in f.seqs]
    assert sorted(covered) == [b.seq for b in blocks], "flow segmentation is not a disjoint cover"
    assert len(covered) == len(set(covered)), "a transition landed in >1 flow"
    # order: prefetch/load, lds_read, compute, epilogue, copy, standalone; A before B within a role
    order = {"prefetch": 0, "load": 0, "lds_read": 1, "compute": 2, "epilogue": 3, "copy": 4, "standalone": 5}
    flows.sort(key=lambda f: (order.get(f.role, 9), f.lane, f.repr_seq))
    return flows


# --------------------------------------------------------------------------------------------------
# Generic flow RENDERER -- one entry point for ANY flow: dispatch a recognized shape to its committed
# (physically gated) builder, else build a generic linear chain from the recorded transitions.
# --------------------------------------------------------------------------------------------------


def _render_linear_flow(pipeline: Any, flow: Flow, out_path: str) -> str:
    """Generic fallback for a memory<->memory chain with no richer recognized shape (a ``copy`` / direct
    ``load``): render each recorded TRANSACTION in the flow as its fragment's register file, joined by
    ``describe_edge``-classified arrows. Algorithm-agnostic -- dims come from the descriptor (``d0``/``d1``),
    dtype from the recorded transaction. (Structural: exercised by a synthetic copy; a real non-MMA kernel
    will refine the per-space component choice -- migration marker.)"""
    from .layout_render import FlowStage, Pipeline, RegisterFileComponent
    from ..transforms import as_forward_map, describe_edge

    txns = [t for t in pipeline.transactions if getattr(t, "seq", None) in set(flow.seqs)]
    txns.sort(key=lambda t: t.seq)
    if not txns:
        raise NotImplementedError(f"flow {flow.role}/{flow.lane} has no recorded transactions to render")
    dims = ("d0", "d1")
    stages, prev = [], None
    for t in txns:
        fm = as_forward_map(t.tile_desc.layout)
        dtype_bits = _elem_bytes(t.dtype_name) * 8
        transform = ""
        if prev is not None:
            kind, why = describe_edge(prev, fm, src_dims=dims, tgt_dims=dims)
            transform = f"{t.space} {t.kind}" if kind == "identity" else f"{t.space} {t.kind} — {kind}\n[{why}]"
        stages.append(FlowStage(f"{t.space} {t.kind} ({t.dtype_name})",
                                RegisterFileComponent(fwd_map=fm, dims=dims, dtype_bits=dtype_bits),
                                source=f"txn[{t.seq}].tile_desc", transform=transform, dist=t.tile_desc.layout))
        prev = fm
    return Pipeline(stages=tuple(stages),
                    title=f"{flow.role} flow {flow.lane}").render_panels(out_path)


def render_flow(pipeline: Any, flow: Flow, out_path: str, *, scope: str = "wave", wave: int = 0,
                buffer: int = 0) -> str:
    """Render ONE segmented :class:`Flow` to ``out_path``. A recognized shape dispatches to its committed,
    physically gated builder (prefetch/lds_read/compute/epilogue); a novel memory chain (copy/load) uses the
    generic linear builder. Fail-fast (naming the gap) on an unhandled role -- never a silent mis-render."""
    role, lane = flow.role, flow.lane
    if role == "prefetch":
        return _view_prefetch_flow(pipeline, operand=lane, scope=scope, buffer=buffer, wave=wave,
                                   out_path=out_path)
    if role == "lds_read":
        return _view_compute_physical(pipeline, operand=lane, wave=wave, out_path=out_path)
    if role == "compute":
        return _view_compute_flow(pipeline, lds_view="logical", operand=lane, wave=wave, out_path=out_path)
    if role == "epilogue":
        return _view_epilogue_flow(pipeline, out_path=out_path)
    if role in ("copy", "load", "standalone"):
        return _render_linear_flow(pipeline, flow, out_path)
    raise NotImplementedError(f"render_flow has no builder for role {role!r} (flow {lane}) -- wire it")


# --------------------------------------------------------------------------------------------------
# Sweep ENUMERATOR + driver -- the committed, algorithm-agnostic policy: L0 overview, L1 flows (each
# transition once), L2 inspections REDIRECTED to their own skills.
# --------------------------------------------------------------------------------------------------

# scope-bearing roles (a macro/wave choice is meaningful); everything else renders once (wave-scoped).
_SCOPED_ROLES = ("prefetch",)


@_dataclass(frozen=True)
class FlowSpec:
    """One enumerated sweep entry. ``level`` 0/1/2; ``seq`` orders within a level; ``name`` is the file
    stem. L0 = an overview view (``l0_view``); L1 = a segmented ``flow`` (rendered); L2 = a ``redirect``
    ``(skill, target)`` the sweep prints instead of drawing (coalescing / bank-conflict own their skills)."""
    level: int
    seq: int
    name: str
    flow: Flow | None = None
    l0_view: str | None = None
    redirect: tuple[str, str] | None = None


def plan_flows(pipeline: Any) -> list[FlowSpec]:
    """Enumerate the sweep for ANY recorded kernel: L0 overviews, L1 flows (from :func:`segment_flows`, so
    each transition appears once), L2 inspections as REDIRECTS. No GEMM assumption; L2 rendering is owned by
    the coalescing / `/bank-conflict` skills, so it is surfaced as a redirect, never drawn here."""
    flows = segment_flows(pipeline)
    specs: list[FlowSpec] = [FlowSpec(0, 0, "block_diagram", l0_view="block_diagram")]
    has_compute = any(f.role == "compute" for f in flows)
    has_coop = any(f.role == "prefetch" for f in flows)
    if has_compute and has_coop:                              # localization needs a cooperative macro + a combine
        specs.append(FlowSpec(0, 1, "localization", l0_view="localization"))
    for i, f in enumerate(flows):                             # L1: one per detected flow (dedup is structural)
        name = f.role if f.lane == "C" else f"{f.role}_{f.lane}"
        specs.append(FlowSpec(1, i, name, flow=f))
    seq = 0                                                   # L2: coalescing + bank-conflict, REDIRECTED
    seen: set[tuple[str, str]] = set()                        # dedup a double-buffered load's repeated instances
    for t in pipeline.transactions:
        if t.space == "global" and t.kind in ("load", "store"):
            op = _operand_of(getattr(t, "space_name", ""))
            if (t.kind, op) in seen:
                continue
            seen.add((t.kind, op))
            specs.append(FlowSpec(2, seq, f"coalescing_{t.kind}_{op}",
                                  redirect=("layout-viz coalescing", f"{t.space} {t.kind} {op}")))
            seq += 1
    for sid in pipeline.lds_spaces():
        specs.append(FlowSpec(2, seq, f"bank_conflict_{_operand_of(pipeline.spaces[sid])}",
                              redirect=("/bank-conflict", pipeline.spaces[sid])))
        seq += 1
    return specs


def render_sweep(pipeline: Any, out_dir: str, *, level: int | None = None, scope: str = "both") -> dict:
    """Drive :func:`plan_flows`: render L0 + L1 into ``out_dir`` with ``{level}_{seq}_{name}[_scope]``
    naming; REPORT L2 as redirects (coalescing / `/bank-conflict` own them -- never drawn here). ``level``
    filters to one level; ``scope='both'`` renders macro+wave for scope-bearing flows. Returns a manifest
    ``{'rendered': [paths], 'redirects': [(name, skill, target)]}``. Fail-fast on an unwired flow role."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {"rendered": [], "redirects": []}
    scopes = ("macro", "wave") if scope == "both" else (scope,)
    for spec in plan_flows(pipeline):
        if level is not None and spec.level != level:
            continue
        stem = f"{out_dir}/{spec.level}_{spec.seq}_{spec.name}"
        if spec.level == 0:
            if spec.l0_view == "block_diagram":
                _bd.block_diagram(pipeline, f"{stem}.png", title="pipeline block diagram")
            else:
                view(pipeline, flow=spec.l0_view, out_path=f"{stem}.png")
            manifest["rendered"].append(f"{stem}.png")
        elif spec.level == 1:
            use = scopes if spec.flow.role in _SCOPED_ROLES else ("wave",)
            for sc in use:
                tag = "macro" if sc == "macro" else "w0"
                manifest["rendered"].append(
                    render_flow(pipeline, spec.flow, f"{stem}_{tag}.png", scope=sc, wave=0))
        else:                                                # L2: redirect, do not render
            manifest["redirects"].append((spec.name, spec.redirect[0], spec.redirect[1]))
    return manifest


# --------------------------------------------------------------------------------------------------
# view() -- the selection -> detail dispatcher (pick a block + scope + drivers -> the committed render)
# --------------------------------------------------------------------------------------------------


def _operand_of(name: str) -> str:
    low = (name or "").lower()
    if "lds_a" in low or low == "%a":
        return "A"
    if "lds_b" in low or low == "%b":
        return "B"
    return "C"


def is_cooperative(encoding: Any, wave_size: int) -> bool:
    """A store/load is COOPERATIVE iff its distribution spans MORE than one wave: its ``wave_dist`` splits
    one macro tile across ``n_waves`` waves, so the encoding decomposes across ``n_waves*wave_size`` lanes
    (``_lane_span`` > wave_size). A per-wave access spans exactly ``wave_size``. Derived from the recorded
    descriptor -- never a flag."""
    return _lane_span(encoding) > wave_size


def view(pipeline: Any, *, block: str | None = None, flow: str | None = None, scope: str = "wave",
         operand: str = "A", buffer: int = 0, wave: int = 0, lds_view: str = "logical",
         out_path: str) -> str:
    """Render a selected view. Give EXACTLY ONE of:
      - ``block`` -- a single node's detail ("lds_store" | ...); or
      - ``flow``  -- a memory-source -> memory-sink chain ("prefetch" = global load -> LDS store).
    The caller supplies only SELECTION: block/flow + ``scope`` ("wave"|"macro") + ``operand``/``buffer``/
    ``wave``. ``lds_view`` ("logical" | "physical") is a compute-flow preference: "logical" = the MMA tee
    (thread-tile geometry); "physical" = the interleaved data AS IT SITS IN LDS BANKS -> the MMA-operand
    registers (the wave read -- clearer for teaching interleaved). Every PHYSICAL fact (dtype, arch, wave
    size, bank count, LDS sizes, cooperative-ness) is DERIVED from the recording -- no silent defaults.
    Flows stay wave-scoped by default so the strip is small + navigable (macro is opt-in)."""
    if (block is None) == (flow is None):
        raise ValueError("view() needs EXACTLY ONE of block= or flow=")
    op = operand.upper()
    if flow == "prefetch":
        return _view_prefetch_flow(pipeline, operand=op, scope=scope, buffer=buffer, wave=wave,
                                   out_path=out_path)
    if flow == "compute":
        return _view_compute_flow(pipeline, lds_view=lds_view, operand=op, wave=wave, out_path=out_path)
    if flow == "localization":
        return _view_wave_localization(pipeline, wave=wave, out_path=out_path)
    if flow == "epilogue":
        return _view_epilogue_flow(pipeline, out_path=out_path)
    if block == "lds_store":
        return _view_lds_store(pipeline, operand=op, buffer=buffer, wave=wave, out_path=out_path)
    raise NotImplementedError(f"view() for block={block!r} flow={flow!r} is not wired yet")


def _with_dtype(out_path: str, dtype: str) -> str:
    """Return ``out_path`` UNCHANGED. The dtype is no longer stamped into the filename -- a sweep's config
    (dtypes included) lives ONCE on the self-describing folder name + each image's own title, so a per-file
    ``_f16`` suffix was redundant noise. Kept as a call-site shim so callers stay uniform; ``dtype`` is unused."""
    return out_path


def _prefetch_descs(pipeline, operand, wave_size):
    """The PREFETCH-flow descriptors + derived physical params for one operand (source = global load,
    sink = LDS store). Everything DERIVED from the recording."""
    sid = next(s for s in pipeline.lds_spaces() if _operand_of(pipeline.spaces[s]) == operand)
    store = next(t for t in pipeline.transactions if t.space_id == sid and t.kind == "store")
    gload = next(t for t in pipeline.transactions
                 if t.space == "global" and t.kind == "load" and _operand_of(t.space_name) == operand)
    dtype = store.dtype_name
    eb = _elem_bytes(dtype)
    n_waves = _lane_span(store.tile_desc.layout) // wave_size
    tile_free = int(gload.tile_desc.shape[0])                  # coop load (free, K): the free extent
    dims = ("N", "K") if operand == "B" else ("M", "K")
    return sid, store, gload, dtype, eb, n_waves, tile_free, dims


def _view_prefetch_flow(pipeline, *, operand, scope, buffer, wave, out_path):
    """PREFETCH flow: the memory-source -> memory-sink chain **global load -> LDS store** for one operand,
    rendered as global tile -> registers -> LDS banks. ``scope='wave'`` shows one wave's band (small +
    navigable); ``'macro'`` shows all waves. Every physical param DERIVED from the recording; the dtype goes
    in the title + filename."""
    from .kernel_stages import flow_load_phase

    arch, wave_size = _arch_wave(pipeline)
    sid, store, gload, dtype, eb, n_waves, tile_free, dims = _prefetch_descs(pipeline, operand, wave_size)
    tile_k = int(gload.tile_desc.shape[1])                     # coop load (free, K): the K extent
    coop_free = tile_free // n_waves                           # one wave's free band
    where = f"all {n_waves} waves" if scope == "macro" else f"wave {wave}"
    note = (f"full {tile_free}x{tile_k} {operand} macro tile, all {n_waves} waves" if scope == "macro"
            else f"wave {wave}: {coop_free}x{tile_k} band of the {tile_free}x{tile_k} {operand} macro tile")
    pipe = flow_load_phase(
        load_desc=gload.tile_desc, store_desc=store.tile_desc, dims=dims, nbanks=_ARCH_NBANKS[arch],
        elem_bytes=eb, lds_base_bytes=buffer * tile_free * eb, dest="lds", scope=scope, macro_note=note,
        cooperative=is_cooperative(store.tile_desc.layout, wave_size), n_waves=n_waves,
        wave_size=wave_size, wave=wave, stride=int(store.strides[0]), load_strides=tuple(gload.strides),
        swizzle=store.swizzle or None,
        title=f"PREFETCH flow {operand} [{dtype}]: global load -> LDS store ({where}, buf {buffer})")
    return pipe.render_panels(_with_dtype(out_path, dtype))


def _view_lds_store(pipeline, *, operand, buffer, wave, out_path):
    """Single LDS-STORE block: one wave's registers -> LDS bank placement (``render_lds_store``). A single
    block is inherently wave-scoped; the all-waves cooperative view is the PREFETCH flow instead."""
    _arch, wave_size = _arch_wave(pipeline)
    sid, store, _gl, dtype, _eb, _nw, _tf, dims = _prefetch_descs(pipeline, operand, wave_size)
    tile_k = int(store.tile_desc.shape[0])                     # store desc (K, free): the K extent
    return render_lds_store(pipeline, sid, tile_k=tile_k, dims=dims, wave=wave, buffer=buffer,
                            out_path=_with_dtype(out_path, dtype))


def _lds_load_shade(pipeline, mma_op, operand, op_enc):
    """Per-element LOAD-TRANSACTION group for an operand fed DIRECTLY by an LDS read: ``{(lane,reg): group}``,
    keyed by the operand's physical slots. AXIS-AGNOSTIC: contiguity from the true layout STRIDES (the store's
    addressing, applied to the operand's own coords), capped at the read's REAL load width (its own maximal
    stride-1 run -- e.g. a b64 ds_read2 -> 4 elems). ``None`` when the operand is not a direct LDS->MMA edge."""
    from collections import Counter

    from .layout_render import transactions, vector_transactions
    from .kernel_stages import lds_inputs
    from ..transforms import as_forward_map

    sid = next((s for s in pipeline.lds_spaces() if _operand_of(pipeline.spaces[s]) == operand), None)
    rd = sid is not None and next((t for t in pipeline.transactions if t.space_id == sid and t.kind == "load"), None)
    st = sid is not None and next((t for t in pipeline.transactions if t.space_id == sid and t.kind == "store"), None)
    if not rd or not st or rd.produces not in mma_op.consumes:
        return None
    _, af = lds_inputs(st.tile_desc, stride=int(st.strides[0]), swizzle=st.swizzle or None)   # true layout strides
    _, af_rd = lds_inputs(rd.tile_desc, stride=int(rd.strides[0]), swizzle=rd.swizzle or None)
    rd_sh = transactions(as_forward_map(rd.tile_desc.layout), af_rd)[0]                        # the read's OWN grouping
    rd_run = max(Counter(t for (l, r), t in rd_sh.items() if l == 0).values())                # elems per real load
    eb = _elem_bytes(rd.dtype_name)
    # cap at b128: a descriptor's contiguous run may exceed one ds_read; the hardware splits it (§9.6 ceiling).
    return vector_transactions(as_forward_map(op_enc), af, eb * 8, max_bits=min(128, rd_run * eb * 8))[0]


def _view_compute_flow(pipeline, *, lds_view, operand, wave, out_path):
    """COMPUTE flow (LDS read -> MMA -> C). ``lds_view="logical"`` (default) = the MMA tee built from the
    RECORDED operands (real interleaved A x B -> DERIVED C, thread-tile geometry). ``lds_view="physical"``
    = the interleaved data AS IT SITS IN LDS BANKS -> the MMA-operand registers (the wave READ), clearer for
    teaching interleaved. Source = LDS reads, sink = C store; all dtypes DERIVED."""
    if lds_view == "physical":
        return _view_compute_physical(pipeline, operand=operand, wave=wave, out_path=out_path)

    import os

    from .layout_render import MmaTee

    mma_op = next(o for o in pipeline.ops if o.kind == "mma")
    reads = [t for t in pipeline.transactions if t.space == "lds" and t.kind == "load"]
    cst = next(t for t in pipeline.transactions if t.space == "global" and t.kind == "store")
    dtype = reads[0].dtype_name                                # A/B input dtype (f16)
    ab_bits, c_bits = _elem_bytes(dtype) * 8, _elem_bytes(cst.dtype_name) * 8

    # FLOW the shade from the operand's real PRODUCER (a direct LDS read) via the shared axis-agnostic helper --
    # the tee's A/B colours are the true load-transaction groups, matching the LDS diagram.
    tee = MmaTee(a_enc=mma_op.a_enc, b_enc=mma_op.b_enc, c_enc=mma_op.c_enc, atom_shape=mma_op.atom_shape,
                 a_canon=mma_op.a_canon, b_canon=mma_op.b_canon, c_canon=mma_op.c_canon,
                 a_dtype_bits=ab_bits, b_dtype_bits=ab_bits, c_dtype_bits=c_bits,
                 dims_a=("M", "K"), dims_b=("N", "K"), dims_c=("M", "N"),
                 shade_a=_lds_load_shade(pipeline, mma_op, "A", mma_op.a_enc),
                 shade_b=_lds_load_shade(pipeline, mma_op, "B", mma_op.b_enc),
                 show_static=True, show_logical_inputs=False, in_dtype=dtype, out_dtype=cst.dtype_name)
    root = os.path.splitext(_with_dtype(out_path, dtype))[0]
    return tee.render(os.path.dirname(root) or ".", name=os.path.basename(root),
                      title=f"COMPUTE flow [{dtype}->{cst.dtype_name}]: LDS read A/B -> MMA (A x B) -> C")


def _operand_lds_stages(pipeline, mma_op, operand, wave_size):
    """DETECT (from the recorded graph, no assumptions) which memory stages an MMA operand actually has:
    - ``rd`` : the LDS read whose ``produces`` is in ``mma.consumes`` (a real producer->consumer edge) -> READBACK
    - ``st`` : a ``store`` on that same LDS space                                                       -> LDS STORE
    - ``coop``: ``_lane_span(store) > wave_size`` (store spans >1 wave)                                 -> COOP FETCH
    Returns a dict; any stage the pipeline lacks is simply absent (None/False)."""
    sid = next((s for s in pipeline.lds_spaces() if _operand_of(pipeline.spaces[s]) == operand), None)
    st = rd = None
    if sid is not None:
        st = next((t for t in pipeline.transactions if t.space_id == sid and t.kind == "store"), None)
        rd = next((t for t in pipeline.transactions if t.space_id == sid and t.kind == "load"
                   and t.produces in mma_op.consumes), None)
    coop = st is not None and _lane_span(st.tile_desc.layout) > wave_size
    return {"sid": sid, "st": st, "rd": rd, "coop": coop, "has_store": st is not None, "has_read": rd is not None}


def _wave_free_band(pipeline, mma_op, operand, op_enc, wave, wave_size):
    """DERIVE this wave's macro band for an operand: (free_extent, n_bands, this_band, K-extent). The MMA operand
    convention (NOT an assumption -- it is the operand definition) is axis 0 = the FREE output dim (M for A,
    N for B) and axis 1 = the shared K/contraction dim. ``n_bands`` = macro/wave element-count ratio (frame-
    agnostic); the wave's band = the M/N offset resolved from the read's SYMBOLIC ORIGIN at this wave's tid,
    falling back to the coop-store free position (flagged ``derived``)."""
    from .kernel_stages import coop_forward_map
    from ..transforms import as_forward_map
    stg = _operand_lds_stages(pipeline, mma_op, operand, wave_size)
    wv = as_forward_map(op_enc)
    free_ext = max(c[0] for c in wv.values()) + 1                           # axis 0 = free (M/N)
    wK = max(c[1] for c in wv.values()) + 1                                 # axis 1 = K (shared)
    st = stg["st"]
    if st is None:                                                          # no cooperative store -> no macro sense
        return None
    nw = _lane_span(st.tile_desc.layout) // wave_size
    cm = coop_forward_map(st.tile_desc, n_waves=nw, wave_size=wave_size)
    n_bands = max(1, len(set(cm.values())) // len(set(wv.values())))        # element-count ratio (frame-agnostic)
    band, derived, fetch = 0, False, None
    reg = [c for (t, r), c in cm.items() if wave * wave_size <= t < (wave + 1) * wave_size]
    if reg:                                                                # this wave's COOP-FETCH strip (1/n_waves)
        me = [max(c[i] for c in cm.values()) for i in (0, 1)]
        sp = [max(c[i] for c in reg) - min(c[i] for c in reg) for i in (0, 1)]
        fax = 0 if sp[0] < sp[1] else 1                                    # store-frame free axis = smaller span
        fetch = (min(c[fax] for c in reg), max(c[fax] for c in reg) + 1)   # the wave's fetch strip in macro coords
        band = (fetch[0] * n_bands) // (me[fax] + 1)                       # fallback band from the fetch position
    if stg["rd"] is not None and getattr(stg["rd"], "origin", None):
        try:                                                              # M/N offset at k=0 = the non-zero comp
            res = resolve_origin(stg["rd"].origin, {"k": 0, "tid": wave * wave_size,
                                                    "block_id": {0: 0, 1: 0, 2: 0}})
            band, derived = int(max(res)) // free_ext, True
        except Exception:
            pass
    return {"free_ext": free_ext, "n_bands": n_bands, "band": band % n_bands, "fetch": fetch, "n_waves": nw,
            "derived": derived, "stg": stg, "wK": wK}


def _view_wave_localization(pipeline, *, wave, out_path):
    """LOCALIZATION view (GEMM/MMA-family): generic "what data does this wave touch, in a MACRO-tile sense".
    APPLICABILITY (gated, not assumed): requires a wave-tile MMA AND at least one COOPERATIVE (macro) store.
    Each operand's flow is drawn ONLY for the stages the recorded graph actually has -- coop FETCH iff the store
    is cooperative, LDS iff it stores, READBACK iff an LDS load feeds the MMA. Bands/axes/offsets are DERIVED
    (free axis = the partitioned one; the wave's band from the read's symbolic origin); nothing hardcodes M/N/K."""
    from ._canvas import _plt

    mma_op = next((o for o in pipeline.ops if o.kind == "mma"), None)
    if mma_op is None:
        raise ValueError("localization view requires a wave-tile MMA op -- this pipeline has none")
    _, wave_size = _arch_wave(pipeline)
    A = _wave_free_band(pipeline, mma_op, "A", mma_op.a_enc, wave, wave_size)
    B = _wave_free_band(pipeline, mma_op, "B", mma_op.b_enc, wave, wave_size)
    if not (A and A["stg"]["coop"]) and not (B and B["stg"]["coop"]):
        raise ValueError("localization view requires a COOPERATIVE macro store (lane span > wave size); none found")
    wM, wN, wK = A["free_ext"], B["free_ext"], A["wK"]
    bm_n, bn_n = A["n_bands"], B["n_bands"]
    bm, bn = A["band"], B["band"]
    macroM, macroN = wM * bm_n, wN * bn_n

    plt = _plt()
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(19, 10)); ax.set_axis_off(); ax.set_xlim(0, 20); ax.set_ylim(0, 11)
    arw = dict(arrowstyle="-|>", lw=2.3, color="#333")

    def block(x0, y0, w, h, rows, cols, label, hi):
        cw, ch = w / cols, h / rows
        for r in range(rows):
            for c in range(cols):
                xx, yy = x0 + c * cw, y0 + (rows - 1 - r) * ch
                on = (r, c) == hi
                ax.add_patch(patches.Rectangle((xx, yy), cw, ch, facecolor="#ffd54f" if on else "#f2f2f2",
                             edgecolor="#d81b60" if on else "#b0b0b0", linewidth=2.4 if on else 0.6))
                ax.text(xx + cw / 2, yy + ch / 2, label(r, c), ha="center", va="center",
                        fontsize=7, fontweight="bold" if on else "normal")

    def ldsbox(x, y, txt):
        ax.add_patch(patches.Rectangle((x, y), 1.6, 2.4, facecolor="#8e24aa", edgecolor="k"))
        ax.text(x + 0.8, y + 1.2, txt, ha="center", va="center", color="white", fontweight="bold", fontsize=9,
                rotation=90)

    def operand_row(y, D, dim, other, x_mma_edge):                  # draw ONLY the stages this operand has
        stg = D["stg"]; ext, nb, bd = D["free_ext"], D["n_bands"], D["band"]
        readx = 6.0
        if stg["coop"] and D["fetch"]:                             # COOP FETCH strip (1/n_waves), only if cooperative
            nw, (f0, f1) = D["n_waves"], D["fetch"]
            sidx = f0 // max(1, (ext * nb) // nw)                  # which fetch-strip row (of n_waves)
            block(0.3, y, 2.4, 3.6, nw, 1, lambda r, c: (f"{dim}[{f0}:{f1}]" if r == sidx else ""), (sidx, 0))
            ax.text(1.5, y + 3.9, f"macro {dim} {ext*nb}x{wK}\ncoop FETCH (1/{nw} strip)", ha="center", fontsize=8)
        if stg["has_store"]:
            ldsbox(3.5, y + 0.6, "LDS")
            ax.annotate("", xy=(3.5, y + 1.8), xytext=(2.7, y + 1.8), arrowprops=arw)
        if stg["has_read"]:                                       # READBACK band, localized on the macro
            block(readx, y, 2.6, 3.6, nb, 1, lambda r, c: f"{dim}[{r*ext}:{(r+1)*ext}]\n{other}", (bd, 0))
            ax.text(readx + 1.3, y + 3.9, f"macro {dim}: wave READ (1/{nb})", ha="center", fontsize=8)
            ax.annotate("", xy=(readx, y + 1.8), xytext=(5.1 if stg["has_store"] else 2.7, y + 1.8), arrowprops=arw)
            ax.annotate("", xy=(x_mma_edge, 5.3 + (0.3 if y > 3 else -0.3)),
                        xytext=(readx + 2.6, y + 1.8), arrowprops=arw)   # read (right edge) -> MMA

    operand_row(6.2, A, "M", f"K[0:{wK}]", 9.6)
    operand_row(0.6, B, "N", f"K[0:{wK}]", 9.6)
    ax.add_patch(patches.Rectangle((9.6, 4.4), 1.8, 1.8, facecolor="#1976d2", edgecolor="k", linewidth=1.5))
    ax.text(10.5, 5.3, "MMA", ha="center", va="center", color="white", fontweight="bold", fontsize=13)
    block(12.6, 1.4, 7.0, 8.0, bm_n, bn_n, lambda r, c: f"M[{r*wM}:{(r+1)*wM}]\nN[{c*wN}:{(c+1)*wN}]", (bm, bn))
    ax.text(16.1, 9.7, f"C macro {macroM}x{macroN} (M x N)", ha="center", fontweight="bold", fontsize=10)
    ax.annotate("", xy=(12.6, 5.3), xytext=(11.4, 5.3), arrowprops=arw)
    note = "band DERIVED from read origin" if (A["derived"] and B["derived"]) else "band from coop-store position"
    ax.set_title(f"LOCALIZATION wave {wave}: what this wave touches in the macro tiles  "
                 f"[coop fetch -> LDS -> wave read -> MMA -> C M[{bm*wM}:{(bm+1)*wM}] x N[{bn*wN}:{(bn+1)*wN}]]  "
                 f"({note})", fontsize=10)
    out = _with_dtype(out_path, "f16")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out



def _view_compute_physical(pipeline, *, operand, wave, out_path):
    """Compute flow, PHYSICAL LDS view: the interleaved data AS IT SITS IN LDS BANKS -> the MMA-operand
    registers (the wave READ) -- clearer for teaching interleaved (you see WHAT is read from LDS and how it
    lands in registers). Addressing uses the RECORDED read descriptor (the one ``verify_lds_roundtrip``
    validated); the register labels are the MMA-operand encoding. All physical params DERIVED."""
    from .kernel_stages import flow_lds_load_placement

    arch, wave_size = _arch_wave(pipeline)
    sid = next(s for s in pipeline.lds_spaces() if _operand_of(pipeline.spaces[s]) == operand)
    read = next(t for t in pipeline.transactions if t.space_id == sid and t.kind == "load")
    store = next(t for t in pipeline.transactions if t.space_id == sid and t.kind == "store")
    mma_op = next(o for o in pipeline.ops if o.kind == "mma")
    operand_enc = mma_op.a_enc if operand == "A" else mma_op.b_enc   # the MMA operand the read feeds
    dtype = read.dtype_name
    dims = ("N", "K") if operand == "B" else ("M", "K")
    pipe = flow_lds_load_placement(
        read_desc=read.tile_desc, flow_desc=operand_enc, dims=dims, nbanks=_ARCH_NBANKS[arch],
        elem_bytes=_elem_bytes(dtype), n_waves=1, wave_size=wave_size, cooperative=False, wave=wave,
        stride=int(read.strides[0]), swizzle=read.swizzle or None,
        store_desc=store.tile_desc, store_stride=int(store.strides[0]), store_swizzle=store.swizzle or None,
        title=f"COMPUTE flow {operand} [{dtype}] PHYSICAL: LDS banks -> MMA-operand registers (wave {wave})")
    return pipe.render_panels(_with_dtype(out_path, dtype))


def _view_epilogue_flow(pipeline, *, out_path):
    """EPILOGUE flow (once, at the very end -- so a small dedicated view): the native accumulator C ->
    {direct | in-register reorder | LDS round-trip} -> global store C. ``c_native`` is derived from the tee
    (built from the recorded op); the C-store distribution is the recorded reorder op's target; the branch
    is auto-classified. Recording-only + derived dtypes."""
    from .kernel_stages import classify_epilogue
    from .layout_render import FlowStage, LogicalTileComponent, MmaTee, Pipeline, RegisterFileComponent
    from ..transforms import as_forward_map, reorder_between

    mma_op = next(o for o in pipeline.ops if o.kind == "mma")
    reo = next((o for o in pipeline.ops if o.kind in ("reorder", "cross_lane")), None)
    reads = [t for t in pipeline.transactions if t.space == "lds" and t.kind == "load"]
    cst = next(t for t in pipeline.transactions if t.space == "global" and t.kind == "store")
    ab_bits, c_bits = _elem_bytes(reads[0].dtype_name) * 8, _elem_bytes(cst.dtype_name) * 8
    tee = MmaTee(a_enc=mma_op.a_enc, b_enc=mma_op.b_enc, c_enc=mma_op.c_enc, atom_shape=mma_op.atom_shape,
                 a_canon=mma_op.a_canon, b_canon=mma_op.b_canon, c_canon=mma_op.c_canon,
                 a_dtype_bits=ab_bits, b_dtype_bits=ab_bits, c_dtype_bits=c_bits,
                 dims_a=("M", "K"), dims_b=("N", "K"), dims_c=("M", "N"))
    c_native = tee.c_mapping()                                 # {(lane,reg)->(m,n)} the machine accumulator
    c_store_desc = reo.tgt_enc if reo is not None else None    # the recorded C-store distribution
    branch, _note = classify_epilogue(c_native, c_store_desc)
    if c_store_desc is None or branch not in ("reorder", "cross_lane"):
        # DIRECT epilogue: native C IS the store order -> one register file, no shuffle arrow.
        pipe = Pipeline(stages=(FlowStage("C registers (native == store order)",
                                          RegisterFileComponent(fwd_map=c_native, dims=("M", "N"),
                                                                dtype_bits=c_bits,
                                                                shade_map={k: 0 for k in c_native}),
                                          source="c_native", dist=mma_op.c_enc),),
                        title=f"EPILOGUE flow [{cst.dtype_name}]: C native -> {branch} -> global store C")
        return pipe.render_panels(_with_dtype(out_path, cst.dtype_name))

    store_fm = as_forward_map(c_store_desc)                     # {(lane,reg)->(m,n)} the C-store register order
    from .layout_render import vector_transactions
    # SHADE = transaction time order (§9.6): ONE shade per vectorized access, width from the RECORDED C-store
    # descriptor STRIDES, capped at b128 (`vector_transactions`), NOT from the register index or `vw`. `addr_fn`
    # is the C store's memory order read straight off `cst.strides` (col-major C -> M stride-1; a row-major C
    # would be N stride-1) -- NEVER assumed. So a lane's contiguous run splits into b128 transactions (e.g. a
    # 16-wide f32 M-run = 4 shades), and the SAME store-transaction shade colours all three panels: the STORE
    # panel is banded (monotonic in memory order), the NATIVE panel is SCRAMBLED (each datum tinted by WHERE it
    # lands in the store) -- the scramble-vs-bands IS the within-lane reorder, made visible + physically honest.
    c_addr = lambda m, n, s=tuple(cst.strides): m * s[0] + n * s[1]
    # §9.6: ORDER THE REGISTER AXIS BY MEMORY ADDRESS -- the encoding/vreg index is NOT physical store order.
    # `c_addr` reads the C descriptor's REAL strides (col-major C -> M stride-1; a row-major C -> N stride-1),
    # so this is descriptor-driven and correct for EITHER major. Lane 0 defines the column order (symmetric
    # across lanes). Without this, an M-contiguous (col-major) store renders N-fast and contradicts the ASM.
    nreg = max(r for _l, r in store_fm) + 1
    mem_order = tuple(sorted(range(nreg), key=lambda r: c_addr(*store_fm[(0, r)])))
    store_ts, _n = vector_transactions(store_fm, c_addr, c_bits, order_by="addr")   # store TIME order = by address
    store_shade = store_ts
    tile_shade = {coord: store_ts[(l, r)] for (l, r), coord in store_fm.items()}               # keyed by (m,n)
    # --- Rule 12: the shuffle is a REAL within-lane reorder with INVARIANT labels (a reorder, never a relabel) ---
    assert c_native != store_fm, "epilogue claims a C-shuffle but native == store order (no reorder to show)"
    assert set(c_native.values()) == set(store_fm.values()), "C-shuffle changed the (M,N) label SET -- not a reorder"
    from collections import defaultdict
    per_lane_native, per_lane_store = defaultdict(set), defaultdict(set)
    for (l, _r), c in c_native.items():
        per_lane_native[l].add(c)
    for (l, _r), c in store_fm.items():
        per_lane_store[l].add(c)
    assert per_lane_native == per_lane_store, "C-shuffle moves a datum across LANES -- that is cross_lane, not a reorder"

    # ILLUSTRATE THE WITHIN-LANE REORDER with ONE detailed thread: pick a tid; ALL of that lane's registers are
    # detailed (per-cell (M,N) labels) in BOTH panels, so you read the SAME datum labels reordered native->store
    # (the shuffle into the actual thread tile). Every OTHER lane is one grouped box with its (M,N) RANGE label.
    from .layout_render import CellGroup
    # DERIVE the C-shuffle reorder (native -> store order) generically -- same machinery as the read side.
    _c_shuffle = reorder_between(c_native, store_fm, pack=max(1, 32 // c_bits))
    focus_tid = 0
    def _c_lane_groups(fwd):
        by = {}
        for slot in fwd:                                       # slot = (lane, reg); group by lane (tid)
            by.setdefault(slot[0], []).append(slot)
        return tuple(CellGroup(frozenset(v), "detailed" if lane == focus_tid else "grouped", "")
                     for lane, v in sorted(by.items()))

    stages = [FlowStage(f"C registers (native accumulator, tid {focus_tid} detailed)",
                        # SOURCE panel (the MMA output, no incoming transaction) -> FLAT shade (hue = lane%8):
                        # there is no vectorized-access time order to tint by here.
                        RegisterFileComponent(fwd_map=c_native, dims=("M", "N"), dtype_bits=c_bits,
                                              color_mode="first8", shade_map={k: 0 for k in c_native},
                                              groups=_c_lane_groups(c_native)),
                        source="c_native", dist=mma_op.c_enc),
              # The C-SHUFFLE is an in-register reorder (native accumulator order -> store order); UNIFIED with
              # the read side via ``reorder_between`` (derived label + cost, never hardcoded). It is a reorder
              # INTERMEDIARY -> no box (the distribution lives on the final stored tile).
              FlowStage(f"C registers (STORE ORDER = memory, tid {focus_tid} detailed)",
                        RegisterFileComponent(fwd_map=store_fm, dims=("M", "N"), dtype_bits=c_bits,
                                              color_mode="first8", shade_map=store_shade,
                                              vreg_values=mem_order,   # §9.6: columns in MEMORY order, not vreg index
                                              groups=_c_lane_groups(store_fm)),
                        source="c_store_desc", reorder=True,
                        transform=f"C-shuffle (in-register reorder)\n{_c_shuffle.label}",
                        info=(_c_shuffle.cost,), dist=c_store_desc),
              # 64x64 stored tile: PER-LANE-PATCH grouping keeps it legible. Shade is the SAME b128-capped, stride-
              # driven store-transaction shade (passed explicitly, so it never falls back to the component's
              # uncapped/row-major default) -- the wide run falls on the C store's real stride-1 axis.
              FlowStage("final C tile (stored, per-lane blocks)",
                        LogicalTileComponent(owner_map={coord: slot for slot, coord in store_fm.items()},
                                             dims=("M", "N"), row_coord=0, mode="thread_tile",
                                             label_coords="logical", font_size=6.5, shade_map=tile_shade),
                        source="c_store_desc", transform="global store", dist=c_store_desc)]
    pipe = Pipeline(stages=tuple(stages),
                    title=f"EPILOGUE flow [{cst.dtype_name}]: C native -> {branch} (within-lane, shade=store order) "
                          f"-> global store C")
    # wider gaps so the LOUD transform-name box (transform_fragment() / classify_transform) clears the panels'
    # y-ticks instead of overprinting them.
    return pipe.render_panels(_with_dtype(out_path, cst.dtype_name), gap_in=4.8)
