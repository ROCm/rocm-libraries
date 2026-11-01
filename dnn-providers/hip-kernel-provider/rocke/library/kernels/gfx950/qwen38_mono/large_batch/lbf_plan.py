# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L1: the large-batch framework's host side (task graph, rows buckets, masking, per-edge sync plan, tables, the
host payload model, LB validate and the index-width bounds). No IR here; the kernel side is :mod:`.lbf`.

``experiments/qwen38-true-mono/DESIGN-LB.md`` 4.1 (rows buckets, runtime token tiles, masking), 5 (sync model), 15.6
(WP-L1) and ``large-batch/P-L2.md`` 5 (the measured per-edge default). The stage graphs are P-L2's
(:mod:`.sync_plan`: FlyTape's phase list ``ft`` and DESIGN-LB's list ``lb``, per-token-tile / per-head / per-expert read
sets); this module adds what a framework needs on top of a probe:

* **Rows buckets** ``{8, 16, 32, 64, 128}`` (:data:`BUCKETS`): a graph is planned for the bucket, the live row count is a
  runtime argument. Every unit carries the first token tile of its rows (``tt``; ``-1`` = not token-tiled, always
  live); with ``ntiles = ceil(rows / 16)`` live tiles a unit with ``tt >= ntiles`` is **masked**: it does no work,
  publishes nothing, and every consumer skips it in its wait set and its read set.
* **Per-edge sync plan** (:class:`EdgeSync`): per stage, the incoming edge is a grid barrier or polled flags; the
  measured default (P-L2 5.1) is ``gmask_full(plan, 0.4)`` (grid where a unit reads >= 0.4 of the previous stage,
  per-wave lazy flags elsewhere).
* **Tags per layer**: layer ``l`` of step ``E`` publishes ``tag = E * ltot + l + 1`` (never 0, distinct per layer and
  step), so one binary walks the layers in one launch (``walk="loop"``) or one launch per layer (``walk="launch"``)
  with identical payload bits.
* **Slots**: payload / flag storage per layer (``slots = ltot``) or per layer parity (``slots = 2``); the previous
  occupant of a slot always has tag ``tag - slots`` (the stale-read twin polls for it).
* **LB validate** (:func:`validate`): no cycle (deadlock freedom with all CTAs resident), and every slot reuse ordered
  after every read of the previous occupant (write-after-read) through program order per CTA, grid barriers, flag edges
  and launch boundaries; optional TP exchange slots across ranks; the CCL exchange's own slot schedule is checked with
  :mod:`..ccl.war` (:func:`ccl_war`).
* **Index widths** (:func:`index_bounds`): the largest value of every i32 address expression of the kernel for a plan,
  checked against 2^31 (the IR-level lint is :mod:`.lbf_lint`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sync_plan import BLOCK, COLGROUPS, M32, W_STEP, Plan, edge_fraction, gmask_full, h_unit, make_plan

__all__ = [
    "BUCKETS", "GRID", "UCOLS", "SCOLS", "TT_SHIFT", "EdgeSync", "LBGraph", "bucket_for", "build_graph", "ccl_war",
    "expected_payload", "index_bounds", "layer_tag", "tables", "validate", "unit_tiles",
]  # fmt: skip

BUCKETS = (8, 16, 32, 64, 128)
GRID = 256
UCOLS = 8  # utab: blk0, nblk, rw, dep_off, dep_cnt, tt, nreaders, 0
SCOLS = 4  # stab: u0, units, flags (bit0 grid before, bit2 poll, bit3 next stage polls = lazy publish), grid prefix
TT_SHIFT = 20  # deps word = unit | (tt + 1) << TT_SHIFT  (unit < 2^20, tt + 1 <= 8)
PAY_WORDS = 512


def bucket_for(rows: int) -> int:
    if not 1 <= rows <= BUCKETS[-1]:
        raise ValueError(f"rows {rows} outside 1..{BUCKETS[-1]}")
    return next(b for b in BUCKETS if rows <= b)


def layer_tag(E: int, ltot: int, layer: int) -> int:
    return (E * ltot + layer + 1) & M32


# ------------------------------------------------------------------------------------------------- token tiles


def unit_tiles(p: Plan) -> list[list[int]]:
    """Per stage, per unit: the first token tile of the unit's rows (``-1``: the unit covers every row). Follows
    :mod:`.sync_plan`'s unit constructions by stage name (the ``lb`` merges keep the surviving stage's units)."""
    rows = p.rows
    out = []
    for s in p.stages:
        n, nm = s.units, s.name
        if nm.endswith("_mix_combine"):
            tt = [min(rows - 1, 2 * k) // 16 for k in range(n)]
        elif nm.endswith("_mix_down"):
            TB = 2 if rows == 128 else 1
            tt = [(k % TB) * 4 if TB == 2 else -1 for k in range(n)]
        elif nm.endswith("_mix_reduce"):
            tt = [min(rows - 1, (512 * k) // 324) // 16 for k in range(n)]
        elif nm == "gdn_recur":
            Hv = 48 // p.tp
            tt = [(4 * (k // Hv)) // 16 for k in range(n)]
        elif nm == "gdn_out_reduce":
            tt = [min(rows - 1, (512 * k) // 1280) // 16 for k in range(n)]
        elif nm == "moe_combine":
            tt = [k // COLGROUPS for k in range(n)]
        else:
            tt = [-1] * n
        out.append(tt)
    return out


# ------------------------------------------------------------------------------------------------- sync plan


@dataclass(frozen=True)
class EdgeSync:
    """The per-edge sync plan of a graph: ``mode`` ``mix`` (grid barrier on the stages in ``gmask``, polled flags on
    the rest), ``grid`` (every stage), ``flags`` (no stage: ``tile`` / ``a2a`` poll every edge), ``none``.
    ``thr``: the ``gmask_full`` threshold that made ``gmask`` (for the record)."""

    mode: str = "mix"
    gmask: tuple[int, ...] = ()
    thr: float = 0.4
    nowait: tuple[int, ...] = ()  # validator mutants only: stages whose edge neither barriers nor polls

    def __post_init__(self) -> None:
        if self.mode not in ("mix", "grid", "flags", "none"):
            raise ValueError(f"EdgeSync.mode {self.mode!r}")

    @staticmethod
    def default(p: Plan, thr: float = 0.4) -> EdgeSync:
        """P-L2's measured default: grid where a unit reads >= ``thr`` of the previous stage."""
        return EdgeSync("mix", tuple(gmask_full(p, thr)), thr)

    def grid_stages(self, S: int) -> set[int]:
        if self.mode == "grid":
            out = set(range(S))
        elif self.mode in ("flags", "none"):
            out = set()
        else:
            out = set(self.gmask)
        return out - set(self.nowait)

    def poll_stages(self, S: int) -> set[int]:
        if self.mode in ("flags",):
            out = set(range(S))
        elif self.mode in ("grid", "none"):
            out = set()
        else:
            out = set(range(S)) - set(self.gmask)
        return out - set(self.nowait)


# ------------------------------------------------------------------------------------------------- graph


@dataclass
class LBGraph:
    """A planned rows bucket: P-L2's stage graph plus token tiles. Unit ids are layer-relative, stage-major."""

    plan: Plan
    tt: list[list[int]]
    seed: int = 1

    @property
    def S(self) -> int:
        return self.plan.nstages

    @property
    def N(self) -> int:
        return self.plan.nunits

    @property
    def bucket(self) -> int:
        return self.plan.rows

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.plan.stages]

    def u0(self) -> list[int]:
        return self.plan.unit0()

    def live(self, rows: int) -> np.ndarray:
        """Bool [N]: units live at ``rows`` live rows."""
        if rows > self.bucket:
            raise ValueError(f"rows {rows} > bucket {self.bucket}")
        nt = -(-rows // 16)
        flat = np.asarray([t for st in self.tt for t in st], dtype=np.int64)
        return (flat < 0) | (flat < nt)

    def prev(self, i: int) -> int:
        return (i - 1) % self.S

    def edge_fraction(self) -> list[float]:
        return edge_fraction(self.plan)


def build_graph(tp: int, rows_bucket: int, kind: str = "lb", null: bool = False, seed: int = 1) -> LBGraph:
    if rows_bucket not in BUCKETS:
        raise ValueError(f"bucket {rows_bucket} not in {BUCKETS}")
    p = make_plan(tp, rows_bucket, kind, null=null, seed=seed)
    g = LBGraph(p, unit_tiles(p), seed)
    for st, t in zip(p.stages, g.tt):
        assert len(t) == st.units
        if any(x >= -(-rows_bucket // 16) for x in t):
            raise AssertionError(f"{st.name}: token tile past the bucket")
    return g


# ------------------------------------------------------------------------------------------------- tables


def tables(g: LBGraph, sync: EdgeSync) -> dict:
    """Device tables (int32 numpy): ``stab`` [S][SCOLS], ``utab`` [N][UCOLS], ``deps`` (flat, words with the
    producer's ``tt + 1`` in the high bits), plus ``gpl`` (grid barriers per layer) and ``s0g`` (stage 0's edge is a
    grid barrier). The kernel's barrier index of stage ``i`` at layer offset ``r`` inside a launch is
    ``r * gpl + prefix_i + 1 - s0g`` (the launch's first stage skips its barrier)."""
    p = g.plan
    S, u0 = g.S, g.u0()
    gs, ps = sync.grid_stages(S), sync.poll_stages(S)
    stab = np.zeros((S, SCOLS), dtype=np.int32)
    pref = 0
    for i, s in enumerate(p.stages):
        nxt = (i + 1) % S
        fl = (1 if i in gs else 0) | (4 if i in ps else 0) | (8 if nxt in ps else 0)
        stab[i] = (u0[i], s.units, fl, pref)
        pref += int(i in gs)
    gpl = pref
    flat_tt = [t for st in g.tt for t in st]
    utab = np.zeros((g.N, UCOLS), dtype=np.int32)
    deps = []
    readers = np.zeros(g.N, dtype=np.int32)
    blk = 0
    for i, s in enumerate(p.stages):
        pi = g.prev(i)
        for u in range(s.units):
            dl = [u0[pi] + d for d in s.deps[u]]
            for d in dl:
                readers[d] += 1
            gu = u0[i] + u
            utab[gu] = (blk, s.blocks[u], int(s.rw), len(deps), len(dl), flat_tt[gu], 0, 0)
            deps += [d | ((flat_tt[d] + 1) << TT_SHIFT) for d in dl]
            blk += s.blocks[u]
    utab[:, 6] = readers
    if g.N >= 1 << TT_SHIFT:
        raise ValueError("too many units per layer for the deps word")
    return dict(stab=stab, utab=utab, deps=np.asarray(deps if deps else [0], dtype=np.int32), gpl=gpl,
                s0g=int(0 in gs), blk_layer=blk)


# ------------------------------------------------------------------------------------------------- host payload model


def expected_payload(g: LBGraph, *, ltot: int, layers: range, E: int, rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(base, live)``: ``base`` uint64 [ltot][N] with payload word ``i`` of unit ``u`` of layer ``l`` =
    ``base[l, u] + i * W_STEP (mod 2^32)``; ``live`` bool [N] (masked units write nothing). Layers outside ``layers``
    are not modelled (0). Layer 0's first stage reads nothing; every other first stage reads the previous layer's last
    stage (also across a launch boundary). A consumer's checksum is the u32 sum over its LIVE read set of the 64
    words ``(d & 7) * 64 + i`` of producer ``d``."""
    p = g.plan
    rows = g.bucket if rows is None else rows
    live = g.live(rows)
    t = tables(g, EdgeSync("grid"))
    ut, dp = t["utab"], t["deps"].astype(np.int64) & ((1 << TT_SHIFT) - 1)
    N = g.N
    H = np.zeros((ltot, N), dtype=np.uint64)
    C = np.zeros((ltot, N), dtype=np.uint64)
    u0 = g.u0()
    idx = np.arange(N, dtype=np.uint64)
    for lay in layers:
        tag = layer_tag(E, ltot, lay)
        for i, s in enumerate(p.stages):
            ids = np.arange(u0[i], u0[i] + s.units)
            H[lay, ids] = [h_unit(tag, lay, i, int(u)) for u in ids]
            if lay == 0 and i == 0:
                continue
            src = lay - 1 if i == 0 else lay
            per_d = (64 * (H[src] + 3 * C[src]) + W_STEP * ((idx & 7) * 4096 + 2016)) & M32
            per_d = np.where(live, per_d, 0).astype(np.uint64)
            for u in ids:
                off, cnt = ut[u, 3], ut[u, 4]
                C[lay, u] = int(per_d[dp[off:off + cnt]].sum()) & M32
    base = (H + 3 * C) & M32
    base[:, ~live] = 0
    return base, live


# ------------------------------------------------------------------------------------------------- LB validate


def _closure_nodes(g: LBGraph, sync: EdgeSync, *, layers: int, walk: str, ranks: int = 1, exchange=()):
    """Event graph of ``layers`` layers on every rank: nodes = tasks (rank, layer, stage, unit) + per-CTA stage-entry
    markers + one hub per grid barrier (+ one hub per launch boundary). Returns (preds, task_node dict, readers)."""
    p = g.plan
    S, u0 = g.S, g.u0()
    gs, ps = sync.grid_stages(S), sync.poll_stages(S)
    preds: list[list[int]] = []
    task: dict[tuple, int] = {}

    def node(pr):
        preds.append(list(pr))
        return len(preds) - 1

    xset = set(exchange)
    last = {}  # (rank, cta) -> last node in program order
    for r in range(ranks):
        for c in range(GRID):
            last[(r, c)] = node([])
    # tasks first get ids lazily; dependency preds appended after all tasks exist
    pending = []
    for lay in range(layers):
        for i, s in enumerate(p.stages):
            first_of_launch = (i == 0) and (lay == 0 or walk == "launch")
            if first_of_launch and lay > 0:  # launch boundary: every event of every rank's previous launch precedes
                for r in range(ranks):
                    hub = node([last[(r, c)] for c in range(GRID)])
                    for c in range(GRID):
                        last[(r, c)] = node([hub])
            if i in gs and not first_of_launch:
                for r in range(ranks):
                    hub = node([last[(r, c)] for c in range(GRID)])
                    for c in range(GRID):
                        last[(r, c)] = node([hub])
            for r in range(ranks):
                for u in range(s.units):
                    c = u % GRID
                    k = node([last[(r, c)]])
                    last[(r, c)] = k
                    task[(r, lay, i, u)] = k
                    if i in ps and not first_of_launch:
                        pending.append((k, r, lay, i, u))
    for k, r, lay, i, u in pending:
        pi, src = g.prev(i), (lay - 1 if i == 0 else lay)
        ds = p.stages[i].deps[u]  # tile read sets: a subset of an a2a wait set, so the check is conservative for a2a
        rk = range(ranks) if (p.stages[pi].name in xset) else (r,)
        for d in ds:
            for r2 in rk:
                preds[k].append(task[(r2, src, pi, d)])
    return preds, task


def _ancestors(preds: list[list[int]]) -> list[int]:
    n = len(preds)
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for k, ps in enumerate(preds):
        for q in ps:
            succ[q].append(k)
            indeg[k] += 1
    stack = [k for k in range(n) if indeg[k] == 0]
    order = []
    while stack:
        k = stack.pop()
        order.append(k)
        for s_ in succ[k]:
            indeg[s_] -= 1
            if indeg[s_] == 0:
                stack.append(s_)
    if len(order) != n:
        raise ValueError("cyclic task graph: a wait depends on itself (deadlock)")
    anc = [0] * n
    for k in order:
        a = 0
        for q in preds[k]:
            a |= anc[q] | (1 << q)
        anc[k] = a
    return anc


def validate(g: LBGraph, sync: EdgeSync, *, slots: int, walk: str = "loop", layers: int = 3, ranks: int = 1,
             exchange: tuple[str, ...] = (), xslots: int = 0, strict: bool = True) -> list[str]:
    """LB validate: deadlock freedom (no cycle), every read ordered after its producer (read-after-write), and
    write-after-read safety of every slot reuse.

    Locations: the payload + flag slot ``(rank, layer mod slots, unit)``; for stages in ``exchange`` (TP: the
    producer pushes its lines to every rank) also the exchange slot ``(dst rank, layer mod xslots, unit)``. The writer of
    layer ``l`` must be ordered after every read of the previous occupant (layer ``l - slots``): program order per
    (rank, CTA), grid barriers, flag edges (a consumer after its producers' publishes, all ranks for an exchange edge)
    and launch boundaries (``walk="launch"``). ``sync.mode == "none"`` orders nothing and is refused. Returns the
    violations (raises with ``strict``)."""
    if sync.mode == "none":
        raise ValueError("sync=none orders nothing: no WAR guarantee")
    if slots < 1 or (exchange and xslots < 1):
        raise ValueError("slots >= 1 (and xslots >= 1 with an exchange)")
    p = g.plan
    layers = max(layers, slots + 1, (xslots + 1) if exchange else 0)
    preds, task = _closure_nodes(g, sync, layers=layers, walk=walk, ranks=ranks, exchange=exchange)
    anc = _ancestors(preds)
    bad = []
    S = g.S
    # readers of (rank, layer, stage, unit): the next stage's units whose read set holds it
    readers = [[[] for _ in range(s.units)] for s in p.stages]
    for i, s in enumerate(p.stages):
        for u, ds in enumerate(s.deps):
            for d in ds:
                readers[g.prev(i)][d].append((i, u))
    xset = set(exchange)
    # read-after-write: every read is ordered after its producer's publish (catches a stage that does not wait)
    for r in range(ranks):
        for lay in range(layers):
            for i, s in enumerate(p.stages):
                if lay == 0 and i == 0:
                    continue
                pi, src = g.prev(i), (lay - 1 if i == 0 else lay)
                rk = range(ranks) if p.stages[pi].name in xset else (r,)
                for u, ds in enumerate(s.deps):
                    k = task[(r, lay, i, u)]
                    for d in ds:
                        for r2 in rk:
                            w = task[(r2, src, pi, d)]
                            if not (anc[k] >> w) & 1 and len(bad) <= 50:
                                bad.append(f"rank {r} layer {lay} {s.name}[{u}] reads rank {r2} layer {src} "
                                           f"{p.stages[pi].name}[{d}] without waiting for it (RAW)")
    for r in range(ranks):
        for lay in range(layers):
            for i, s in enumerate(p.stages):
                nxt = (i + 1) % S
                for loc_slots, scope in ((slots, "slot"), (xslots, "xslot")):
                    if scope == "xslot" and s.name not in xset:
                        continue
                    old = lay - loc_slots
                    if old < 0:
                        continue
                    rl = old + 1 if nxt == 0 else old  # the layer of the readers of layer `old`'s stage i
                    if rl >= layers or (nxt == 0 and rl == 0):
                        continue
                    for u in range(s.units):
                        w = task[(r, lay, i, u)]
                        rdr_ranks = range(ranks) if scope == "xslot" else (r,)
                        for (ci, cu) in readers[i][u]:
                            for r2 in rdr_ranks:
                                k = task.get((r2, rl, ci, cu))
                                if k is None:
                                    continue
                                if not (anc[w] >> k) & 1:
                                    bad.append(f"rank {r} layer {lay} {s.name}[{u}] overwrites {scope} {lay % loc_slots} "
                                               f"before rank {r2} layer {rl} {p.stages[ci].name}[{cu}] read layer {old}")
                                    if len(bad) > 50:
                                        break
    if bad and strict:
        raise ValueError(f"LB validate: {len(bad)} RAW / WAR violations, e.g. {bad[:3]}")
    return bad


def ccl_war(spec, **kw) -> list[str]:
    """The CCL exchange's own slot schedule (``ccl.war.spec_schedule`` of a ``ccl.layout`` spec) checked with
    ``ccl.war.check``: the arena-slot half of DESIGN-CCL 4.9's rule; :func:`validate` checks the LB-graph half."""
    from ..ccl import war

    return war.check(war.spec_schedule(spec, **kw))


# ------------------------------------------------------------------------------------------------- index widths


@dataclass
class Bounds:
    """Largest value of every i32 address expression of the :mod:`.lbf` kernel (bytes or elements)."""

    items: dict = field(default_factory=dict)

    def check(self, limit: int = (1 << 31) - 1) -> list[str]:
        return [f"{k}: {v} > {limit}" for k, v in self.items.items() if v > limit]


def index_bounds(g: LBGraph, *, ltot: int, slots: int, pf: int, dly_layers: int | None = None) -> Bounds:
    """The kernel's i32 index expressions at their maxima (the formulas mirror :mod:`.lbf`):

    * payload byte offset ``((slot * N + u) * 512 + tid) * 4`` with ``slot < slots``;
    * flag byte offset ``((slot * N + u) * PF + w) * 4``;
    * delay-table element ``layer * N + u`` (``layer < ltot``);
    * deps element ``dep_off + dep_cnt``; unit table element ``N * UCOLS``;
    * weight block range ``nblk * BLOCK`` (the per-unit buffer resource size, i32) and the weight base is i64
      (``(layer * blk_layer + blk0) * BLOCK`` zero-extended before the multiply)."""
    t = tables(g, EdgeSync("grid"))
    N = g.N
    ut = t["utab"]
    b = Bounds()
    b.items["pay_byte"] = ((slots * N - 1) * PAY_WORDS + PAY_WORDS - 1) * 4 + 3
    b.items["flg_byte"] = ((slots * N - 1) * pf + pf - 1) * 4 + 3
    b.items["dly_elem"] = (dly_layers or ltot) * N
    b.items["deps_elem"] = int((ut[:, 3] + ut[:, 4]).max())
    b.items["utab_elem"] = N * UCOLS
    b.items["wblk_bytes"] = int(ut[:, 1].max()) * BLOCK
    b.items["blk_layer_x_ltot"] = 0  # i64 path (zext before the multiply): recorded as 0 by construction
    b.items["unit_id"] = N
    return b
