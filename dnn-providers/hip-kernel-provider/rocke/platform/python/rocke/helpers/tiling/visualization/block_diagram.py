# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Level-0 pipeline BLOCK DIAGRAM -- the map you navigate before drilling into a stage.

Turns a :class:`~rocke.helpers.tiling.tiling_recorder.RecordedPipeline` into a labelled boxes-and-arrows
figure: one block per recorded node -- a mem<->vreg TRANSACTION (global/LDS load/store, the accumulator
fill) or a vreg<->vreg OP (``reorder``/``cross_lane`` transform, the ``mma``). Edges follow the MEMORY
producer -> consumer, NOT mere program order. The K-loop body sits in a loop container (a structured
``scf.for`` emits it once, so ``xN`` is inherent); loop membership is detected from the origin DAG (a node
whose ``window.origin`` references the ``scf.for`` induction variable is inside the loop).

The loop body holds TWO INDEPENDENT flows, decoupled by the double buffer, drawn as parallel lanes:
  * COMPUTE (this tile):   LDS read (cur buf) -> MMA -> accumulator
  * PREFETCH (next tile):  global load (k+1) -> LDS store (other buf)
The prefetch load feeds the STORE, never the read; the read consumes a buffer written a PREVIOUS iteration
(the dashed "swap buffers" edge). Drawing them as one linear chain would imply a dependency that isn't there.

This is the entry point of the selection flow: read the diagram, pick a block, then the coordinator asks
scope (macro/wave) + drivers (wave/buffer/operand) and renders that block's detail via the Phase-C driver.
No layout judgement lives here -- it only reflects the recorded structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _canvas as cv

# ---- combining ops: the register-level COMBINE that anchors a "compute" flow -----------------------
# Generic (algorithm-agnostic): a compute block is any op that fuses register operands into a result. MMA
# is today's only instance; extend this tuple as `dot`/`reduce`/`fma`/... land so a non-GEMM kernel's
# compute is detected the same way. Everything that (transitively) feeds a combining op is compute; the
# rest of the loop is prefetch. A kernel with NO combining op simply has no compute flow (correct).
COMBINING_OPS = ("mma",)

# ---- logical AXIS NAMES -- derived, never assumed -------------------------------------------------
# A recording carries NO axis names: `strides` / `tile_desc.shape` are positional. The only two sources
# are the lane tag (itself derived from the recorded space name) and the driver's own convention
# (`auto_pipeline` already writes ("N","K") for B, ("M","K") for A, ("M","N") for C). So the named dims
# are used ONLY when the recording actually contains a COMBINING op -- `_operand` falls back to "C" for
# anything unmatched, so on a non-combining recording naming its axes "M"/"N" would be an invention.
_COMBINE_DIMS = {"A": ("M", "K"), "B": ("N", "K"), "C": ("M", "N")}
_POSITIONAL_DIMS = ("d0", "d1")

# ---- lane / operand colours (A = blue, B = green, C/accumulator = orange) ------------------------
_FILL = {"A": "#cfe3f7", "B": "#d7eecf", "C": "#f7e2c9"}
_EDGE = "#333333"
_BROWN = "#8a6d3b"

# ---- geometry (data units; y grows DOWN, axis inverted) -----------------------------------------
_PITCH = 1.5       # vertical distance between successive rows
_BOX_H = 0.92      # block height
_W_WIDE, _W_LOOP = 2.6, 1.7
_XW = {"A": 1.9, "B": 5.7, "C": 3.8}      # single-column phases (prologue / epilogue)
_XC = {"A": 1.35, "B": 3.15, "C": 2.25}   # loop COMPUTE lane  (this tile: read -> mma)
_XP = {"A": 4.85, "B": 6.65, "C": 5.75}   # loop PREFETCH lane (next tile: load -> store)


# --------------------------------------------------------------------------------------------------
# Structure extraction (recording -> ordered blocks + loop span)
# --------------------------------------------------------------------------------------------------


def _depends_on_iv(value: Any) -> bool:
    """True if the SSA ``value`` DAG references the ``scf.for`` induction variable (the K-loop iter)."""
    if isinstance(value, int):
        return False
    op = getattr(value, "op", None)
    if op is None:
        return False
    if op.name == "scf.for":
        return True
    return any(_depends_on_iv(o) for o in getattr(op, "operands", ()))


def _origin_uses_iv(origin: Any) -> bool:
    return bool(origin) and any(_depends_on_iv(v) for v in origin)


def _operand(node: Any) -> str:
    """Which logical operand lane a node sits on (A / B / C), from the node ALONE. The accumulator fill
    and the combining op genuinely ride C. A ``reorder``/``cross_lane`` does NOT -- it inherits the lane
    of whatever produced the value it consumes, which only the dataflow graph knows; this returns the "C"
    default and :func:`_resolve_reg_lanes` overrides it. (Forcing every transform onto C is what labelled
    the A/B operand bridges as C-epilogue work.)"""
    if node.kind in ("fill", "mma"):
        return "C"
    name = (getattr(node, "space_name", "") or "").lower()
    if "lds_a" in name or name == "%a":
        return "A"
    if "lds_b" in name or name == "%b":
        return "B"
    return "C"


def _value_edges(nodes: list) -> list[tuple[int, int]]:
    """Producer -> consumer edges by SSA Value identity, straight off the recorded nodes."""
    producer = {n.produces: n.seq for n in nodes if getattr(n, "produces", None) is not None}
    return [(producer[c], n.seq) for n in nodes for c in getattr(n, "consumes", ())
            if c in producer and producer[c] != n.seq]


def forward_reach(edges: list[tuple[int, int]]):
    """``reach(seq) -> set`` of nodes reachable FORWARD (producer -> consumer) from ``seq``. ONE home --
    the flow segmenter in ``auto_pipeline`` delegates here rather than keeping a second copy."""
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


def rmw_spaces(nodes: list) -> set[int]:
    """The memory spaces this recording BOTH globally loads and globally stores -- i.e. read-modify-write
    buffers (the epilogue re-reads its own output tile). ONE home: the block-diagram caption and the flow
    segmenter both route on this same recorded fact, never on a GEMM noun like "beta*C"."""
    glob = [n for n in nodes if getattr(n, "space", "") == "global"]
    loaded = {getattr(n, "space_id", None) for n in glob if n.kind == "load"}
    stored = {getattr(n, "space_id", None) for n in glob if n.kind == "store"}
    return {s for s in loaded & stored if s is not None}


def same_distribution(enc_a: Any, enc_b: Any) -> bool:
    """Do two distributions place data identically? Compared by FORWARD MAP (slot -> coord), never by
    encoding identity/equality: structurally different encodings can place data the same way, and the tile
    SHAPE lives on the TileDesc, not the encoding -- so raw encoding equality answers a different question."""
    if enc_a is None or enc_b is None:
        return False
    from ..transforms import as_forward_map
    try:
        return as_forward_map(enc_a) == as_forward_map(enc_b)
    except Exception:
        return False


def _resolve_reg_lanes(nodes: list, lanes: dict[int, str]) -> None:
    """In-place: give every ``reorder``/``cross_lane`` the lane of the node that PRODUCED the value it
    consumes. An LDS/global read on an operand space tags A/B; a combining op or the fill tags C. Nodes are
    walked in recorded order, so a chain (read -> reorder -> reorder) resolves transitively."""
    producer = {n.produces: n.seq for n in nodes if getattr(n, "produces", None) is not None}
    for n in nodes:
        if n.kind not in ("reorder", "cross_lane"):
            continue
        src = next((producer[c] for c in getattr(n, "consumes", ()) if c in producer), None)
        if src is not None:
            lanes[n.seq] = lanes[src]


def _stride1_caption(node: Any, dims: tuple[str, ...]) -> str:
    """Name the tensor's stride-1 (CONTIGUOUS) axis, DERIVED from the recorded strides.

    ✗ NEVER a "row-major"/"col-major" word: the major word points at a DIFFERENT physical axis for each of
    A / B / C, which is exactly how a hardcoded ``"col-major (M-contig)"`` survived on an N-stride-1 C. The
    stride-1 axis is the whole fact -- say only that. ✗ Never "coalesced" either: contiguity is a per-tensor
    stride fact, coalescing is a wave-level lane-major property and is not this diagram's verdict to make.
    If the stride-1 axis is not unique (no extent>1 axis at stride 1, or several), state the RAW recorded
    strides rather than guessing one."""
    strides = tuple(getattr(node, "strides", None) or ())
    shape = getattr(getattr(node, "tile_desc", None), "shape", None)
    hits = [i for i, s in enumerate(strides)
            if isinstance(s, int) and s == 1 and (shape is None or int(shape[i]) > 1)]
    if len(hits) == 1 and hits[0] < len(dims):
        return f"{dims[hits[0]]} stride-1 (contiguous)"
    return f"strides {strides}"


def _reg_op_detail(node: Any, lane: str, *, src: Any, feeds_combine: bool, to_store_order: bool) -> str:
    """The italic second line of a transform block, DERIVED from what the transform actually BRIDGES (its
    recorded producer; whether it feeds a combining op; whether it lands in the recorded store's
    distribution) -- never from its node kind. Captioning every transform "(C epilogue)" labelled the A/B
    operand bridges as C work.

    Kept SHORT and in the same "-> destination" grammar as the neighbouring blocks ("-> other buf",
    "<- cur buf"): the loop lane's boxes are 1.7 units wide, and a fuller phrase overprints its neighbour.
    The long form ("operand bridge -- the price of the wide read") belongs on the flow panel title, where
    there is room."""
    if lane in ("A", "B") and getattr(src, "space", None) in ("lds", "global") and feeds_combine:
        return "-> MMA order"
    if src is not None and src.kind in COMBINING_OPS:
        return "-> store order" if to_store_order else "from MMA acc"
    return ""                            # endpoints not both recognised -> claim nothing


def _label(node: Any, lane: str) -> str:
    """The block's bold caption: the node and the lane it rides, in one consistent grammar
    (``LDS read A`` / ``reorder A`` / ``global store C``). The transform's ROLE is the italic
    detail line (:func:`_reg_op_detail`), not part of the name."""
    k = node.kind
    if k == "fill":
        return f"fill {lane} acc\n= 0"
    if k == "mma":
        return "MMA\nA x B -> C"
    if k in ("reorder", "cross_lane"):
        return f"{'reorder' if k == 'reorder' else 'cross-lane'} {lane}"
    sp = getattr(node, "space", "")
    verb = {("global", "load"): "global load", ("global", "store"): "global store",
            ("lds", "load"): "LDS read", ("lds", "store"): "LDS store"}.get((sp, k), k)
    return f"{verb} {lane}"


def _sublabel(node: Any, phase: str, *, dims: tuple[str, ...] = _POSITIONAL_DIMS,
              stages_to_lds: bool = False, is_rmw: bool = False, reg_detail: str = "") -> str:
    """The block's italic second line -- ONE line (it is drawn as a single label inside a 0.92-high box).

    A global LOAD is a staging PREFETCH only if its value actually reaches an LDS store (the POSITIVE test
    comes first: "does not reach an LDS store" is vacuously true for a load whose consumer was never
    recorded, so a negative test alone would silently mislabel a recording gap). Otherwise, if its space is
    also globally stored, it is the read-modify-write input. Otherwise: say nothing -- silence is honest,
    an invented "prefetch k=0" on an epilogue re-read is not."""
    sp, k = getattr(node, "space", ""), node.kind
    if k in ("reorder", "cross_lane"):
        return reg_detail
    if sp == "lds" and k == "store":
        return "-> buf 0" if phase == "prologue" else "-> other buf"
    if sp == "lds" and k == "load":
        return "<- cur buf" if phase == "loop" else "<- last buf"
    if sp == "global" and k == "load":
        if stages_to_lds:
            return "prefetch k+1" if phase == "loop" else "prefetch k=0"
        return "read-modify-write input" if is_rmw else ""
    if sp == "global" and k == "store":
        return _stride1_caption(node, dims)
    return ""


@dataclass(frozen=True)
class Block:
    seq: int
    kind: str
    space: str
    lane: str
    phase: str      # prologue | loop | epilogue
    label: str
    sublabel: str
    produces: int | None = None       # id() of the SSA Value this node yields
    consumes: tuple[int, ...] = ()     # id()s of the SSA Values this node reads
    space_id: int | None = None        # identity of the memory space (a global buffer that is BOTH
                                       # loaded and stored is a read-modify-write target)


def extract_blocks(pipeline: Any) -> tuple[list[Block], int | None, int | None]:
    """Ordered blocks + the loop-body seq span ``(lo, hi)`` (``None`` if the pipeline has no K-loop).

    Lanes and captions are DERIVED from the recorded dataflow graph, not from node kind: a transform takes
    the lane of its producer, a global load is a prefetch only if it reaches an LDS store, a global store
    names its own stride-1 axis. Axis NAMES are used only when the recording has a combining op."""
    nodes = list(pipeline.nodes)
    loop_seqs = [n.seq for n in nodes if _origin_uses_iv(getattr(n, "origin", None))]
    lo = min(loop_seqs) if loop_seqs else None
    hi = max(loop_seqs) if loop_seqs else None

    by_seq = {n.seq: n for n in nodes}
    edges = _value_edges(nodes)
    reach = forward_reach(edges)
    producer = {n.produces: n.seq for n in nodes if getattr(n, "produces", None) is not None}
    combine = {n.seq for n in nodes if n.kind in COMBINING_OPS}
    lds_stores = {n.seq for n in nodes if getattr(n, "space", "") == "lds" and n.kind == "store"}
    rmw = rmw_spaces(nodes)
    gstore = next((n for n in nodes
                   if getattr(n, "space", "") == "global" and n.kind == "store"), None)
    store_enc = getattr(getattr(gstore, "tile_desc", None), "layout", None)

    lanes = {n.seq: _operand(n) for n in nodes}
    _resolve_reg_lanes(nodes, lanes)

    blocks: list[Block] = []
    for n in nodes:
        if lo is None:
            phase = "loop"
        elif n.seq < lo:
            phase = "prologue"
        elif n.seq <= hi:
            phase = "loop"
        else:
            phase = "epilogue"
        lane = lanes[n.seq]
        dims = _COMBINE_DIMS.get(lane, _POSITIONAL_DIMS) if combine else _POSITIONAL_DIMS
        src = by_seq.get(next((producer[c] for c in getattr(n, "consumes", ()) if c in producer), None))
        detail = _reg_op_detail(n, lane, src=src, feeds_combine=bool(reach(n.seq) & combine),
                                to_store_order=same_distribution(getattr(n, "tgt_enc", None), store_enc))
        label = _label(n, lane)
        sublabel = _sublabel(n, phase, dims=dims, reg_detail=detail,
                             stages_to_lds=bool(reach(n.seq) & lds_stores),
                             is_rmw=getattr(n, "space_id", None) in rmw)
        blocks.append(Block(seq=n.seq, kind=n.kind, space=getattr(n, "space", "reg"), lane=lane,
                            phase=phase, label=label, sublabel=sublabel,
                            produces=getattr(n, "produces", None),
                            consumes=tuple(getattr(n, "consumes", ())),
                            space_id=getattr(n, "space_id", None)))
    return blocks, lo, hi


# --------------------------------------------------------------------------------------------------
# Dataflow (edges + lanes) -- DERIVED from the recorded SSA Value graph, not from kind/lane heuristics
# --------------------------------------------------------------------------------------------------


def _dataflow_edges(blocks: list[Block]) -> list[tuple[int, int]]:
    """Real producer -> consumer edges by SSA Value identity: an edge ``(p, c)`` exists iff node ``c``
    consumes a Value that node ``p`` produced. This is the recorded data dependency graph -- no layout or
    kind assumptions. (The one dependency it CANNOT see is the accumulator carried across the ``scf.for``
    iter-arg boundary, which rebinds the Value -- reconstructed separately by :func:`_acc_bridge`.)"""
    return _value_edges(blocks)


def _loop_lanes(blocks: list[Block], edges: list[tuple[int, int]]) -> dict[int, str]:
    """Split the loop body into its independent flows FROM THE GRAPH (not a kind rule): a loop node is
    COMPUTE if it transitively feeds a loop COMBINING op (``COMBINING_OPS`` -- mma today, extensible), else
    PREFETCH (it loads/stores a future tile)."""
    loop = {b.seq for b in blocks if b.phase == "loop"}
    pred: dict[int, list[int]] = {}
    for f, t in edges:
        pred.setdefault(t, []).append(f)
    compute = {b.seq for b in blocks if b.phase == "loop" and b.kind in COMBINING_OPS}
    stack = list(compute)
    while stack:
        for p in pred.get(stack.pop(), []):
            if p in loop and p not in compute:
                compute.add(p)
                stack.append(p)
    return {s: ("compute" if s in compute else "prefetch") for s in loop}


def _acc_bridge(blocks: list[Block]) -> list[tuple[int, int]]:
    """The accumulator carry crosses the ``scf.for`` iter-arg boundary, which REBINDS the SSA Value, so it
    cannot be chained by Value identity. Reconstruct it by pairing the dangling register-path produces
    (fill / loop-MMA output, consumed only by the loop op) with the dangling consumes (each MMA's acc
    input, produced only by the loop op) in program order -- the single accumulator thread."""
    reg = {"fill", "mma", "reorder", "cross_lane"}
    produced = {b.produces for b in blocks if b.produces is not None}
    consumed = {c for b in blocks for c in b.consumes}
    outs = sorted(b.seq for b in blocks if b.kind in reg and b.produces is not None
                  and b.produces not in consumed)
    ins = sorted(b.seq for b in blocks if b.kind in reg and any(c not in produced for c in b.consumes))
    return list(zip(outs, ins))


# --------------------------------------------------------------------------------------------------
# Layout + render
# --------------------------------------------------------------------------------------------------


@dataclass
class Placed:
    block: Block
    xc: float       # box centre x
    y: float        # box top y
    w: float
    h: float = _BOX_H


def _stack(blocks: list[Block], xc_of, w: float, y0: float) -> tuple[list[Placed], float]:
    """Place ``blocks`` top-down, pairing consecutive same-kind A/B into one row; a single block centres
    on lane C. Returns (placements, y_end)."""
    placed: list[Placed] = []
    r = i = 0
    while i < len(blocks):
        b = blocks[i]
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        y = y0 + r * _PITCH
        if b.lane == "A" and nxt and nxt.lane == "B" and nxt.kind == b.kind:
            placed.append(Placed(b, xc_of("A"), y, w))
            placed.append(Placed(nxt, xc_of("B"), y, w))
            i += 2
        else:
            placed.append(Placed(b, xc_of(b.lane if b.lane in ("A", "B") else "C"), y, w))
            i += 1
        r += 1
    return placed, y0 + r * _PITCH


def _layout(blocks: list[Block], lanes: dict[int, str]) -> tuple[list[Placed], tuple[float, float] | None]:
    """Prologue + epilogue are single wide columns; the loop body is TWO parallel lanes -- COMPUTE (left)
    and PREFETCH (right), per the graph-derived ``lanes`` -- sharing a y-span. Returns
    (placed, (loop_y0, loop_y1) | None)."""
    pre = [b for b in blocks if b.phase == "prologue"]
    loop = [b for b in blocks if b.phase == "loop"]
    epi = [b for b in blocks if b.phase == "epilogue"]

    placed, y = [], 0.0
    pp, y = _stack(pre, lambda l: _XW[l], _W_WIDE, y)
    placed += pp

    loop_span = None
    if loop:
        y0 = y + 0.5
        comp = [b for b in loop if lanes.get(b.seq) == "compute"]
        pref = [b for b in loop if lanes.get(b.seq) == "prefetch"]
        cp, yc = _stack(comp, lambda l: _XC[l], _W_LOOP, y0)
        fp, yf = _stack(pref, lambda l: _XP[l], _W_LOOP, y0)
        placed += cp + fp
        y = max(yc, yf)
        loop_span = (y0, y)

    ep, _ = _stack(epi, lambda l: _XW[l], _W_WIDE, y + 0.5)
    placed += ep
    return placed, loop_span


def _draw_block(ax, p: Placed):
    x0 = p.xc - p.w / 2.0
    cv.fill(ax, x0, p.y, _FILL[p.block.lane], w=p.w, h=p.h, edge="none")
    cv.box(ax, x0, p.y, p.w, p.h, color=_EDGE, lw=1.6, zorder=4)
    fs = 8.8 if p.w < 2.0 else 10.0
    cv.label(ax, p.xc, p.y + p.h * 0.40, p.block.label, fs=fs, weight="bold", zorder=5)
    if p.block.sublabel:
        cv.label(ax, p.xc, p.y + p.h * 0.80, p.block.sublabel, fs=7, style="italic", color="#555", zorder=5)


def _draw_edge(ax, pf: Placed, pt: Placed, style: str, label: str, color: str):
    ls = "-" if style == "solid" else (0, (4, 3))
    x1, y1 = pf.xc, pf.y + pf.h                        # tail = producer bottom
    if pt.y >= y1:                                     # forward (down)
        ax.annotate("", xy=(pt.xc, pt.y), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color, linestyle=ls,
                                    mutation_scale=13), zorder=2)
        mx, my = (x1 + pt.xc) / 2, (y1 + pt.y) / 2
    else:                                              # back-edge (up): curve to the consumer's bottom
        ax.annotate("", xy=(pt.xc, pt.y + pt.h), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color=color, linestyle=ls,
                                    connectionstyle="arc3,rad=0.45", mutation_scale=13), zorder=2)
        mx, my = (x1 + pt.xc) / 2 - 0.55, (y1 + pt.y + pt.h) / 2
    if label:
        ax.text(mx, my, label, fontsize=7, style="italic", color=color, ha="center", va="center",
                zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def _kernel_meta(pipeline: Any) -> str:
    """Best-effort ONE-LINE kernel summary DERIVED from the recording (never guessed): arch/wave, dtype
    in->out, atom, wave tile, macro tile + wave grid, and features (cooperative / double-buffered). Any piece
    that can't be derived is silently omitted -- the line degrades gracefully, never fabricates."""
    from ..transforms import as_forward_map
    p = []
    arch = getattr(pipeline, "arch", None); ws = getattr(pipeline, "wave_size", None)
    if arch: p.append(str(arch))
    if ws: p.append(f"wave{ws}")
    txns = list(pipeline.transactions)
    din = sorted({t.dtype_name for t in txns if t.space == "lds" and t.kind == "load"})
    dout = sorted({t.dtype_name for t in txns if t.space == "global" and t.kind == "store"})
    if din and dout: p.append(f"{'/'.join(din)}\u2192{'/'.join(dout)}")
    wm = wn = wk = None
    mma = next((o for o in pipeline.ops if o.kind == "mma"), None)
    if mma is not None:
        if mma.atom_shape: p.append("atom " + "\u00d7".join(map(str, mma.atom_shape)))
        try:
            c, a = as_forward_map(mma.c_enc), as_forward_map(mma.a_enc)
            wm = max(v[0] for v in c.values()) + 1; wn = max(v[1] for v in c.values()) + 1
            wk = max(v[1] for v in a.values()) + 1
            s = f"wave {wm}\u00d7{wn}\u00d7{wk}"
            if mma.atom_count: s += f" (\u00d7{mma.atom_count} atoms)"
            p.append(s)
        except Exception:
            pass
    def _span(enc):                                   # threads an encoding spans (ALL lane-partition levels)
        s = 1
        for majors, minors in zip(enc.lane_to_rh_major, enc.lane_to_rh_minor):
            for maj, mnr in zip(majors, minors):
                s *= enc.bucket_length(maj, mnr)
        return s
    def _macro_free(opl):                             # free extent = (cooperative-store elements) / macro-K.
        for t in txns:                                # convention-free: NO shape-slot / free-axis assumption;
            if t.space == "lds" and t.kind == "store" and opl in pipeline.spaces.get(t.space_id, "").lower():
                try:                                  # the coop fetch covers one K-slice of depth = wave K.
                    return (_span(t.tile_desc.layout) * t.register_count) // wk
                except Exception:
                    return None
        return None
    mm = _macro_free("_a") if wk else None
    mn = _macro_free("_b") if wk else None
    if mm and mn:
        p.append(f"macro {mm}\u00d7{mn}\u00d7{wk}")
        if wm and wn:
            p.append(f"{mm // wm}\u00d7{mn // wn} waves")
    feats = []
    stores = {}
    for t in txns:
        if t.space == "lds" and t.kind == "store":
            k = pipeline.spaces.get(t.space_id, ""); stores[k] = stores.get(k, 0) + 1
    if stores and max(stores.values()) > 1: feats.append("double-buffered")
    if mm and wm and mn and wn and (mm // wm) * (mn // wn) > 1: feats.append("cooperative")
    if feats: p.append("(" + ", ".join(feats) + ")")
    return "   \u00b7   ".join(p)


def block_diagram(pipeline: Any, out_path: str, *, title: str = "") -> str:
    """Render the Level-0 block diagram of ``pipeline`` to ``out_path`` (PNG). Returns the path. A one-line
    kernel summary (arch, dtype, atom, wave/macro tile, features) is DERIVED from the recording + shown as a
    subtitle."""
    blocks, lo, _hi = extract_blocks(pipeline)
    edges = _dataflow_edges(blocks)          # real producer->consumer, from the SSA Value graph
    lanes = _loop_lanes(blocks, edges)       # compute/prefetch split by reachability to the MMA
    acc = _acc_bridge(blocks)                # the accumulator carry across the scf.for boundary
    placed, loop_span = _layout(blocks, lanes)
    by_seq = {p.block.seq: p for p in placed}
    plt = cv._plt()

    y_end = max(p.y + p.h for p in placed)
    fig = plt.figure(figsize=(8.2, max(6.0, (y_end + 1.4) * 0.6)))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.93])
    ax.set_xlim(-2.0, 8.7)
    ax.set_ylim(y_end + 0.6, -1.2)
    ax.axis("off")

    # loop container + two-lane headers + divider (behind the blocks)
    if loop_span is not None:
        y0, y1 = loop_span
        cv.box(ax, 0.15, y0 - 0.72, 7.55, (y1 + 0.35) - (y0 - 0.72), color=_BROWN, lw=1.6, zorder=1)
        cv.label(ax, 0.35, y0 - 0.70, "K-loop  x N tiles  (double-buffered)", fs=9.5, weight="bold",
                 color=_BROWN, ha="left", va="top")
        cv.label(ax, _XC["C"], y0 - 0.30, "COMPUTE  (this tile)", fs=8.5, weight="bold",
                 color="#2b6cb0", ha="center")
        cv.label(ax, _XP["C"], y0 - 0.30, "PREFETCH  (next tile)", fs=8.5, weight="bold",
                 color="#2f855a", ha="center")
        ax.plot([4.0, 4.0], [y0 - 0.12, y1 + 0.12], color="#c8c8c8", lw=1.0, ls=(0, (3, 3)), zorder=1)
        cv.label(ax, 3.9, y1 - 0.28,
                 "double buffer: PREFETCH parks tile k+1 in the OTHER buffer;\n"
                 "next iteration's COMPUTE reads it (the two buffers swap every tile)",
                 fs=7.5, style="italic", color=_BROWN, ha="center", va="center", zorder=6)

    for fseq, tseq in edges:                                       # solid = real Value dataflow
        if fseq in by_seq and tseq in by_seq:
            _draw_edge(ax, by_seq[fseq], by_seq[tseq], "solid", "", _EDGE)
    for fseq, tseq in acc:                                          # dashed = accumulator carry (scf.for-bridged)
        if fseq in by_seq and tseq in by_seq:
            _draw_edge(ax, by_seq[fseq], by_seq[tseq], "dashed", "acc", _BROWN)
    for p in placed:
        _draw_block(ax, p)

    for ph, col in (("prologue", "#555"), ("loop", _BROWN), ("epilogue", "#555")):
        ys = [p.y for p in placed if p.block.phase == ph]
        if ys:
            cv.label(ax, -1.8, (min(ys) + max(ys) + _BOX_H) / 2, ph.upper(), fs=10, weight="bold",
                     color=col, rotation=90, ha="center")

    ax.set_title(title or "pipeline block diagram", fontsize=13, weight="bold", pad=20)
    meta = _kernel_meta(pipeline)
    if meta:
        ax.annotate(meta, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#555")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
