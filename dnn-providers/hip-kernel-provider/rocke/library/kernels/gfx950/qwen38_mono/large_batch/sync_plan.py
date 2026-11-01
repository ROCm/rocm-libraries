# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L2: the stage graph of one large-batch decoder layer for the sync-model chain (host only, no IR).

``experiments/qwen38-true-mono/DESIGN-LB.md`` 5 and 12 (P-L2). A layer is a list of stages; a stage is a list of
units, each with the bytes it streams (8 KiB blocks, the stage's per-rank HBM bytes from :mod:`.bytes_model` spread
over its units), whether it writes them back (``rw``: the GDN state), and its **read set**: the units of the
previous stage (stage 0: the last stage of the previous layer) whose output it reads. The read set is the data
dependency and is the same for every sync arm; the arms differ only in what a unit waits for (``sync_chain.py``):

* ``grid``  a striped grid barrier before every stage (FlyTape's phases),
* ``tile``  the producers of its read set (per-token-tile, per-head, per-expert polled edges),
* ``a2a``   every unit of the previous stage (the mailbox all-to-all: no partial-edge knowledge, no barrier),
* ``none``  nothing (timing floor only: its numerics are racy by construction).

Plans
-----
``ft``: FlyTape's tiled tape phase list (``flytape/models/qwen4/model.py``, ``bodies_tiled.py``): per GDN layer at TP1
the 16 phases ``attn_mix_combine, attn_mix_down, attn_mix_reduce, attn_mix_up, gdn_proj, gdn_recur, gdn_out,
gdn_out_reduce, mlp_mix_combine .. mlp_mix_up, moe_router, moe_up, moe_down, moe_combine``; at TP2 (per rank, the
exchange itself not modelled: WP-C5 measures it) 14 (no ``gdn_out_reduce``: the out-proj fold is in the phase; the
MoE is one fused ``moe_expert`` phase). Unit counts follow FlyTape's plans (``MIX_SPLITS`` 40 / 20 x 2 token blocks,
``OUT_SPLITS`` 6, ``GDN_SPLITS`` 2 + the ragged one-chunk units, 64-row groups, the router's 9 row groups); the
grid-strided phases (combine, reduce) and the recurrence use the granularity noted per stage below (the recurrence
as (4 rows, value head) units instead of FlyTape's (row, head) workgroups, so per-head flags stay few).

``lb``: DESIGN-LB 4.2's stage list: ``ft`` with the HC combine merged into the down projection, the split-K reduces
merged into their producer (last-arriver folds: the fold bytes are spread over the producer units, so a fold's own
latency on the critical path is NOT modelled), ``gdn_out_reduce`` merged into ``gdn_out``: 11 stages at TP1, 10 at TP2.

``null=True`` keeps every graph and zeroes every unit's bytes (pure hand-off cost).

Routing: uniform random top-10 of 512 per row (FlyTape's random-weight workload), one draw per plan (every layer of
a launch uses it, so every layer has the same units); the shared expert is unit ``U`` of the expert stage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .bytes_model import step_bytes

BLOCK = 8192  # bytes one CTA streams per step (512 lanes x 16 B)
TOPK = 10
EXPERTS = 512
COLGROUPS = 10  # moe_combine column groups (256 hidden columns each)
HIDDEN = 2560
HC = 4
ROWG = 64  # weight rows per FlyTape row group
PLANS = ("ft", "lb")


@dataclass
class Stage:
    name: str
    blocks: list[int]  # per unit
    deps: list[list[int]]  # per unit: local unit indices of the previous stage
    rw: bool = False
    note: str = ""

    @property
    def units(self) -> int:
        return len(self.blocks)


@dataclass
class Plan:
    tp: int
    rows: int
    kind: str
    null: bool
    stages: list[Stage] = field(default_factory=list)
    experts: list[int] = field(default_factory=list)  # active routed experts (sorted)
    route: list[list[int]] = field(default_factory=list)  # per row: its 10 experts

    # ---------------------------------------------------------------- flat tables
    @property
    def nstages(self) -> int:
        return len(self.stages)

    def unit0(self) -> list[int]:
        out, n = [], 0
        for s in self.stages:
            out.append(n)
            n += s.units
        return out

    @property
    def nunits(self) -> int:
        return sum(s.units for s in self.stages)

    def blocks_per_layer(self) -> int:
        return sum(sum(s.blocks) for s in self.stages)

    def bytes_per_layer(self) -> dict[str, float]:
        return {s.name: sum(s.blocks) * BLOCK * (2 if s.rw else 1) for s in self.stages}

    def check(self) -> None:
        S = self.nstages
        for i, s in enumerate(self.stages):
            prev = self.stages[(i - 1) % S]
            if len(s.deps) != s.units:
                raise ValueError(f"{s.name}: {len(s.deps)} dep lists for {s.units} units")
            for u, d in enumerate(s.deps):
                if not d:
                    raise ValueError(f"{s.name} unit {u}: empty read set")
                if len(set(d)) != len(d) or min(d) < 0 or max(d) >= prev.units:
                    raise ValueError(f"{s.name} unit {u}: bad read set {d[:8]}.. (prev {prev.name} has {prev.units})")
            if any(b < 0 for b in s.blocks):
                raise ValueError(f"{s.name}: negative blocks")


def _blocks(total_bytes: float, units: int) -> list[int]:
    """``total_bytes`` spread over ``units`` as whole 8 KiB blocks (rounded up per unit, at least one)."""
    per = max(1, -(-int(round(total_bytes)) // (units * BLOCK)))
    return [per] * units


def _tiles(rows: int) -> int:
    return -(-rows // 16)


def _route(rows: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed * 7919 + rows)
    return [sorted(rng.sample(range(EXPERTS), TOPK)) for _ in range(rows)]


# ------------------------------------------------------------------------- FlyTape's phase list


def _mix(kind: str, rows: int, by: dict, prev_rows_of_unit, prev_units_for_rows) -> list[Stage]:
    """combine -> down -> reduce -> up of one HC mixer (``tiled_mix``)."""
    # combine: a wave per (row, stream), 8 waves a CTA unit -> units of 2 rows x 4 streams; reads the previous stage's
    # output of all columns of its rows; bytes: residual r (4 streams), y and the bf16 dst written (3 x 2560 x 2 x 4)
    ncomb = -(-rows * HC // 8)
    comb_rows = [list(range(2 * k, min(rows, 2 * k + 2))) for k in range(ncomb)]
    comb = Stage(f"{kind}_mix_combine", _blocks(rows * HC * HIDDEN * 2 * 3, ncomb),
                 [prev_units_for_rows(r) for r in comb_rows], note="2 rows x 4 streams per unit")
    # down: 6 row groups (324 rows) x S splits (x 2 token blocks at B128) over K = 4 x 2560
    G = -(-324 // ROWG)
    TB = 2 if rows == 128 else 1
    S = 40 // TB
    tbr = 64 if TB == 2 else rows
    down_deps = []
    for g in range(G):
        for s in range(S):
            for tb in range(TB):
                r = list(range(tb * tbr, min(rows, (tb + 1) * tbr)))
                down_deps.append(sorted({k for k in range(ncomb) if set(comb_rows[k]) & set(r)}))
    down = Stage(f"{kind}_mix_down", _blocks(by["hc_down"], G * S * TB), down_deps,
                 note=f"{G} row groups x {S} splits x {TB} token blocks")
    # reduce: (token tile, row group) units; each folds the S splits of its row group and token block
    # reduce: FlyTape's element-strided fold (``_mix_reduce``: one thread per (row, output row), 512 a worker): unit k
    # = elements [512 k, 512 k + 512) of the rows x 324 outputs; it reads the S split partials of its elements
    nred = -(-rows * 324 // 512)
    red_deps = []
    for kk in range(nred):
        e0, e1 = 512 * kk, min(rows * 324, 512 * kk + 512)
        gs = sorted({(e % 324) // ROWG for e in range(e0, e1)})
        tbs = sorted({(e // 324) // tbr for e in range(e0, e1)})
        red_deps.append(sorted((g * S + s_) * TB + tb for g in gs for s_ in range(S) for tb in tbs))
    red = Stage(f"{kind}_mix_reduce", _blocks(S * rows * 324 * 4, nred), red_deps,
                note="FlyTape's element-strided fold: 512 (row, output) elements per unit, S partials each")
    # up: 160 row groups (10240 rows), N = every row: reads all of `down` (every token, every row group)
    up = Stage(f"{kind}_mix_up", _blocks(by["hc_up"], 160), [list(range(nred))] * 160,
               note="160 row groups, all rows: a full edge")
    return [comb, down, red, up]


def _x_slice_deps(lo: int, hi: int) -> list[int]:
    """mix_up row groups (of 160) holding the x columns [lo, hi) of all 4 streams."""
    return [r for r in range(160) if (r % 40) * ROWG < hi and (r % 40) * ROWG + ROWG > lo]


def ft_stages(tp: int, rows: int, seed: int = 1) -> tuple[list[Stage], list[int], list[list[int]]]:
    sb = step_bytes(tp=tp, rows=rows, regime="matched")
    st = {s.name: s for s in sb.stages}
    T = _tiles(rows)
    by = {"hc_down": st["hc_attn_down"].hbm, "hc_up": st["hc_attn_up"].hbm}
    route = _route(rows, seed)
    experts = sorted({e for r in route for e in r})
    U = len(experts)
    Hv = 48 // tp
    Hk = 16 // tp
    out_units = 40 * 6

    # The layer's last stage (moe_combine): (token tile, column group) units; its unit index helper is needed by the
    # first stage of the layer (prev = the previous layer's moe_combine).
    def comb_units_for_rows(rr):
        ts = sorted({r // 16 for r in rr})
        return [t * COLGROUPS + c for t in ts for c in range(COLGROUPS)]

    stages = _mix("attn", rows, by, None, comb_units_for_rows)

    # gdn_proj: whole 64-row groups x GDN_SPLITS 2 (+ the ragged rest as one-chunk units)
    nrows = (2 * Hk * 128) + 2 * (Hv * 128) + 2 * Hv  # q, k, v, z, b, a
    whole = nrows // ROWG
    rag = nrows - whole * ROWG
    pdeps = []
    for g in range(whole):
        for s in range(2):
            pdeps.append(_x_slice_deps(s * 1280, (s + 1) * 1280))
    nrag = 20 if rag else 0
    for c in range(nrag):
        pdeps.append(_x_slice_deps(c * 128, (c + 1) * 128))
    gbytes = st["gdn_in"].hbm
    proj_blocks = _blocks(gbytes * (whole * ROWG) / nrows, whole * 2) + (_blocks(gbytes * rag / nrows, nrag) if nrag else [])
    proj = Stage("gdn_proj", proj_blocks, pdeps, note=f"{whole} groups x 2 splits + {nrag} ragged one-chunk units")

    # gdn_recur: (4 rows, value head) units; reads the proj units holding head h's q / k (key head h * Hk / Hv), v, z, b, a
    def proj_units_of_rows(r0, r1):
        out = set()
        for r in range(r0, r1):
            g = r // ROWG
            if g < whole:
                out.update((2 * g, 2 * g + 1))
            else:
                out.update(range(2 * whole, 2 * whole + nrag))
        return out

    quads = -(-rows // 4)
    rdeps = []
    for q in range(quads):
        for h in range(Hv):
            kh = h * Hk // Hv
            d = set()
            d |= proj_units_of_rows(kh * 128, kh * 128 + 128)
            d |= proj_units_of_rows(Hk * 128 + kh * 128, Hk * 128 + kh * 128 + 128)
            v0 = 2 * Hk * 128
            d |= proj_units_of_rows(v0 + h * 128, v0 + h * 128 + 128)
            z0 = v0 + Hv * 128
            d |= proj_units_of_rows(z0 + h * 128, z0 + h * 128 + 128)
            b0 = z0 + Hv * 128
            d |= proj_units_of_rows(b0 + h, b0 + h + 1)
            d |= proj_units_of_rows(b0 + Hv + h, b0 + Hv + h + 1)
            rdeps.append(sorted(d))
    state = st["gdn_recur"].state  # read + write
    recur = Stage("gdn_recur", _blocks(state / 2, quads * Hv), rdeps, rw=True,
                  note="(4 rows, value head) units, f32 state read + written")

    # gdn_out: 40 row groups x OUT_SPLITS 6, split s = heads [s Hv / 6, (s + 1) Hv / 6)
    odeps = []
    for g in range(40):
        for s in range(6):
            hs = range(s * Hv // 6, (s + 1) * Hv // 6)
            odeps.append(sorted(q * Hv + h for q in range(quads) for h in hs))
    out = Stage("gdn_out", _blocks(st["gdn_out"].hbm, out_units), odeps, note="40 row groups x 6 head splits")
    stages += [proj, recur, out]
    if tp == 1:
        # FlyTape's element-strided out reduce: unit k = column pairs [512 k, 512 k + 512) of rows x 1280 pairs,
        # reading the 6 split partials of its columns' row groups
        nored = -(-rows * (HIDDEN // 2) // 512)
        odeps2 = []
        for kk in range(nored):
            p0, p1 = 512 * kk, min(rows * (HIDDEN // 2), 512 * kk + 512)
            gs = sorted({((pp % (HIDDEN // 2)) * 2) // ROWG for pp in range(p0, p1)})
            odeps2.append(sorted(g * 6 + s_ for g in gs for s_ in range(6)))
        ored = Stage("gdn_out_reduce", _blocks(6 * rows * HIDDEN * 4, nored), odeps2,
                     note="FlyTape's element-strided fold: 512 column pairs per unit, 6 partials each")
        stages.append(ored)

        def y_units_for_rows(rr):
            return sorted({kk for r in rr for kk in range(r * (HIDDEN // 2) // 512, (r * (HIDDEN // 2) + HIDDEN // 2 - 1) // 512 + 1)})
    else:
        def y_units_for_rows(rr):
            return list(range(out_units))  # the in-phase fold: every (row group, split) unit of the rows' columns

    stages += _mix("mlp", rows, by, None, y_units_for_rows)

    # moe_router: 9 row groups (513 rows) x 20 K splits of 128 columns
    rt_deps = [_x_slice_deps(s * 128, (s + 1) * 128) for g in range(9) for s in range(20)]
    router = Stage("moe_router", _blocks(st["router"].hbm, 180), rt_deps, note="9 row groups x 20 splits")
    stages.append(router)

    # experts: routed (active) + the shared expert as unit U; parts so the grid fills
    nexp = U + 1
    eb = st["experts"].hbm / U
    sb_ = st["shared_expert"].hbm
    if tp == 1:
        parts = max(1, -(-256 // nexp))
        up_blocks, dn_blocks = [], []
        for e in range(nexp):
            byt = sb_ if e == U else eb
            up_blocks += _blocks(byt * 2 / 3 / parts, 1) * parts
            dn_blocks += _blocks(byt / 3 / parts, 1) * parts
        all_rt = list(range(180))
        up = Stage("moe_up", up_blocks, [all_rt] * (nexp * parts), note=f"{nexp} experts x {parts} parts; route: full edge")
        dn = Stage("moe_down", dn_blocks, [[e * parts + p for p in range(parts)] for e in range(nexp) for _ in range(parts)],
                   note="per-expert edge (all up parts of the expert)")
        stages += [up, dn]
        last_parts = parts
    else:
        parts = max(1, -(-256 // nexp))
        xb = []
        for e in range(nexp):
            xb += _blocks((sb_ if e == U else eb) / parts, 1) * parts
        ex = Stage("moe_expert", xb, [list(range(180))] * (nexp * parts),
                   note=f"{nexp} experts x {parts} parts (gate_up + down fused); route: full edge")
        stages.append(ex)
        last_parts = parts

    # moe_combine: (token tile, column group); reads the parts covering its columns of every expert its rows use
    eidx = {e: i for i, e in enumerate(experts)}
    cdeps = []
    for t in range(T):
        used = sorted({eidx[e] for r in range(t * 16, min(rows, t * 16 + 16)) for e in route[r]} | {U})
        for c in range(COLGROUPS):
            lo, hi = c * 256, (c + 1) * 256
            ps = [p for p in range(last_parts) if p * HIDDEN // last_parts < hi and (p + 1) * HIDDEN // last_parts > lo]
            cdeps.append(sorted(e * last_parts + p for e in used for p in ps))
    comb = Stage("moe_combine", _blocks(rows * (TOPK + 1) * HIDDEN * 4, T * COLGROUPS), cdeps,
                 note="(token tile, 256 columns); per-expert partial edge")
    stages.append(comb)
    return stages, experts, route


# ------------------------------------------------------------------------- transforms


def merge(stages: list[Stage], idx: int, into: str) -> list[Stage]:
    """Remove stage ``idx``: its consumers read what it read (union of read sets); its bytes go to the producer
    (``into="prev"``, a last-arriver fold) or the consumer (``into="next"``, a fused prologue)."""
    S = len(stages)
    rem = stages[idx]
    prev_i, next_i = (idx - 1) % S, (idx + 1) % S
    nxt = stages[next_i]
    new_deps = [sorted({p for d in deps for p in rem.deps[d]}) for deps in nxt.deps]
    total = sum(rem.blocks)
    out = []
    for i, s in enumerate(stages):
        if i == idx:
            continue
        blocks, deps = list(s.blocks), s.deps
        if i == next_i:
            deps = new_deps
        tgt = prev_i if into == "prev" else next_i
        if i == tgt and s.units:
            add = -(-total // s.units)
            blocks = [b + add for b in blocks]
        out.append(Stage(s.name, blocks, deps, s.rw, s.note))
    return out


def lb_stages(stages: list[Stage]) -> list[Stage]:
    """``ft`` -> DESIGN-LB's list: combine into down, reduces into their producer."""
    def at(st, name):
        return next(i for i, s in enumerate(st) if s.name == name)

    st = stages
    for kind in ("attn", "mlp"):
        st = merge(st, at(st, f"{kind}_mix_reduce"), "prev")
        st = merge(st, at(st, f"{kind}_mix_combine"), "next")
    if any(s.name == "gdn_out_reduce" for s in st):
        st = merge(st, at(st, "gdn_out_reduce"), "prev")
    return st


def make_plan(tp: int, rows: int, kind: str = "ft", null: bool = False, seed: int = 1) -> Plan:
    if kind not in PLANS:
        raise ValueError(kind)
    stages, experts, route = ft_stages(tp, rows, seed)
    if kind == "lb":
        stages = lb_stages(stages)
    if null:
        stages = [Stage(s.name, [0] * s.units, s.deps, s.rw, s.note) for s in stages]
    p = Plan(tp, rows, kind, null, stages, experts, route)
    p.check()
    return p


# ------------------------------------------------------------------------- device tables

UCOLS = 8  # utab columns: blk0, nblk, rw, dep_off, dep_cnt, prev_u0, prev_n, stage
TCOLS = 4  # task columns: layer, stage, unit (-1: dummy), flags (bit 0 grid sync before, bit 1 no wait, sync index << 8)
GRID = 256


def unit_table(p: Plan) -> tuple[list[int], list[int]]:
    """(utab flat [N * UCOLS], deps flat): unit ids are layer-relative (stage-major)."""
    u0 = p.unit0()
    S = p.nstages
    utab, deps = [], []
    blk = 0
    for i, s in enumerate(p.stages):
        pi = (i - 1) % S
        for u in range(s.units):
            dl = [u0[pi] + d for d in s.deps[u]]
            utab += [blk, s.blocks[u], int(s.rw), len(deps), len(dl), u0[pi], p.stages[pi].units, i]
            deps += dl
            blk += s.blocks[u]
    return utab, deps


def task_table(p: Plan, layers: int, gmask=None) -> tuple[list[int], int]:
    """Per CTA (``GRID``): ``[count, (layer, stage, unit, flags) x Tmax]`` flat, and ``Tmax``. Unit ``k`` of a stage
    runs on CTA ``k mod GRID`` (FlyTape's ``range(worker, units, workers)``); a CTA without a unit in a stage gets a
    dummy task (unit -1) so the grid arm's barrier sees every CTA. Sync index = ``layer * S + stage`` (none for the
    launch's first stage, whose units also skip the wait: they read the previous launch's last layer).

    ``gmask`` (``sync="mix"``): the stage indices whose incoming edge is a grid barrier; the other stages' tasks get
    bit 2 (poll the read set's flags) and the sync index counts grid boundaries only; bit 3 marks tasks whose next
    stage polls (the mix arm's lazy publish). ``None``: every stage."""
    u0 = p.unit0()
    S = p.nstages
    per = [[] for _ in range(GRID)]
    gm = set(range(S)) if gmask is None else set(gmask)
    ordn = 0
    for lay in range(layers):
        for i, s in enumerate(p.stages):
            k = lay * S + i
            g = k > 0 and i in gm
            ordn += int(g)
            nxt_polls = ((i + 1) % S) not in gm  # bit 3: the next stage polls this stage's flags (mix lazy publish)
            fl = (1 if g else 0) | (2 if k == 0 else 0) | (0 if i in gm else 4) | (8 if nxt_polls else 0)
            fl |= (ordn if g else 0) << 8
            for c in range(GRID):
                mine = list(range(c, s.units, GRID))
                if not mine:
                    per[c].append((lay, i, -1, fl))
                for j, u in enumerate(mine):
                    per[c].append((lay, i, u0[i] + u, fl if j == 0 else (fl & 14)))
    tmax = max(len(x) for x in per)
    flat = []
    for c in range(GRID):
        row = [len(per[c])]
        for t in per[c]:
            row += list(t)
        row += [0] * (TCOLS * (tmax - len(per[c])))
        flat += row
    return flat, tmax


def nsync(p: Plan, layers: int, gmask=None) -> int:
    """Grid barriers per launch."""
    if gmask is None:
        return layers * p.nstages - 1
    gm = set(gmask)
    return sum(1 for lay in range(layers) for i in range(p.nstages) if (lay or i) and i in gm)


def edge_fraction(p: Plan) -> list[float]:
    """Per stage: the mean share of the previous stage's units a unit reads (1.0 = an all-to-all edge)."""
    S = p.nstages
    return [sum(len(d) for d in s.deps) / (s.units * p.stages[(i - 1) % S].units) for i, s in enumerate(p.stages)]


def gmask_full(p: Plan, thr: float) -> list[int]:
    """Stages whose edge reads at least ``thr`` of the previous stage on average (grid there, flags elsewhere)."""
    return [i for i, f in enumerate(edge_fraction(p)) if f >= thr]


# ------------------------------------------------------------------------- host model of the payload

M32 = 0xFFFFFFFF
H_TAG, H_L, H_S, H_U = 0x01000193, 0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D
W_STEP = 0x9E37


def h_unit(tag: int, layer: int, stage: int, unit: int) -> int:
    return (tag * H_TAG + layer * H_L + stage * H_S + unit * H_U) & M32


def expected_checksums(p: Plan, layers: int, tag: int):
    """Per (layer, unit): (H, chk). Payload word ``i`` of unit ``u`` = ``H + 3 chk + i W_STEP``; a consumer's chk =
    sum over its read set ``d`` of the 64 words ``(d & 7) * 64 + i`` of ``d``'s payload (the words producer wave
    ``d & 7`` stored): ``64 (H_d + 3 chk_d) + W_STEP ((d & 7) 4096 + 2016)``. The launch's first stage reads nothing
    (chk 0)."""
    import numpy as np

    utab, deps = unit_table(p)
    N = p.nunits
    ut = np.asarray(utab, dtype=np.int64).reshape(N, UCOLS)
    dp = np.asarray(deps, dtype=np.int64)
    H = np.zeros((layers, N), dtype=np.uint64)
    C = np.zeros((layers, N), dtype=np.uint64)
    u0 = p.unit0()
    for lay in range(layers):
        for i, s in enumerate(p.stages):
            ids = np.arange(u0[i], u0[i] + s.units)
            H[lay, ids] = [(h_unit(tag, lay, i, int(u))) for u in ids]
            if lay == 0 and i == 0:
                continue
            src = lay - 1 if i == 0 else lay
            per_d = (64 * (H[src] + 3 * C[src]) + W_STEP * ((np.arange(N, dtype=np.uint64) & 7) * 4096 + 2016)) & M32
            for u in ids:
                off, cnt = ut[u, 3], ut[u, 4]
                C[lay, u] = int(per_d[dp[off:off + cnt]].sum()) & M32
    return H, C
