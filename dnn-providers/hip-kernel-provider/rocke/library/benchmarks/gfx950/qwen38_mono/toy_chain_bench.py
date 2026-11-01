# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device harness and CLI for the layer-kernel framework's toy chain.

Torch-free (``rocke.runtime.hip_module`` through the persistent-mailbox
example harness). Every timing is a device ``s_memrealtime`` tick difference
or a hipEvent interval over a hipGraph; every run checks numerics first.

Commands (``python -m benchmarks.gfx950.qwen38_mono.toy_chain_bench <cmd>``
with the rocKE ``library/`` and ``platform/python`` on ``sys.path``):

* ``check``: numerics of a knob grid (every y / r mailbox, tags, h_out,
  checksums, counters), single launches and 2-layer chains vs stacks;
* ``replay``: ``--replays`` graph replays with an advancing step counter
  (the T9 gate: 10^4), checked against the host model at the end;
* ``time``: the depth sweep and knob arms, interleaved (ABAB...), stamped
  builds decoded per stage; raw ticks to ``--out``, ratios to stdout.

Absolute numbers go only to the ``--out`` JSON (keep it out of the repo).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import sys
from dataclasses import dataclass

import numpy as np

from kernels.gfx950.qwen38_mono import mailbox_ir  # noqa: F401  (registers impl="ir")
from kernels.gfx950.qwen38_mono import stamps as stp
from kernels.gfx950.qwen38_mono import toy_chain as T
from kernels.gfx950.qwen38_mono.framework import ATOM_LAYER_ARGS, FrameworkKnobs
from rocke.examples.gfx950.persistent_mailbox import harness as H
from rocke.helpers import mailbox as mbx

ARCH = "gfx950"
ARG_FMT = "<QiI" + "Q" * len(ATOM_LAYER_ARGS)
H_WORDS = 64  # each h buffer: 64 i32 words (h_out[4 li], [4 li + 1])


def pack_args(step: int, layer: int, **addrs: int) -> bytes:
    vals = [addrs.get(a, 0) for a in ATOM_LAYER_ARGS]
    return struct.pack(ARG_FMT, step, layer, 0, *vals)


def wrap32(v: np.ndarray | int):
    if isinstance(v, np.ndarray):
        return ((v.astype(np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int64)
    return (int(v) + (1 << 31)) % (1 << 32) - (1 << 31)


@dataclass
class ToyBufs:
    spec: T.ToyChainSpec
    scratch: int
    scratch_bytes: int
    step: int
    h: list[int]
    w: list[int]  # weight copies
    x: int
    w_host: np.ndarray
    x_host: np.ndarray
    seed: int
    extra: dict | None = None  # more kernel args (e.g. the NUMA flip table)


def scratch_alloc(dev: H.Dev, spec: T.ToyChainSpec) -> tuple[int, int, dict]:
    """Scratch for ``spec`` (2 MiB aligned: NUMA copies need 16 KiB and the
    IOD calibration works per 2 MiB page) and the extra args: with
    ``knobs.numa_flip_arg``, a calibrated flip per region."""
    lay = T.toy_layout(spec)
    nbytes = lay.total_bytes(spec.knobs.stamps)
    raw = dev.alloc(nbytes + (2 << 20))
    scratch = (raw + (2 << 20) - 1) & ~((2 << 20) - 1)
    extra = {}
    arg = spec.knobs.numa_flip_arg
    if arg:
        flags = H.calibrate_iod_home(dev, ARCH, scratch, lay.mailbox_bytes)
        tab = np.zeros(lay.table_entries, np.int32)
        for name, _ in lay.regions:
            for slot in range(lay.parity_slots):
                for row in range(lay.rows):
                    off = lay.offset(name, row, slot)
                    tab[lay.region_index(name, row, slot)] = flags[off >> 21]
        extra[arg] = dev.put(tab)
    return scratch, nbytes, extra


def alloc(dev: H.Dev, spec: T.ToyChainSpec, *, chain: int = 1, copies: int = 1,
          seed: int = 12345, rng_seed: int = 7) -> ToyBufs:  # fmt: skip
    """Buffers for ``chain`` consecutive launches (each ``spec.layers``
    layers) and ``copies`` weight copies (cold timing rotates them)."""
    scratch, nbytes, extra = scratch_alloc(dev, spec)
    step = dev.alloc(256)
    h = [dev.alloc(H_WORDS * 4) for _ in range(chain + 1)]
    dev.write(h[0], seed_words(spec, seed))
    rng = np.random.default_rng(rng_seed)
    rw = max(spec.row_words, 1)
    w_host = rng.integers(-1000, 1000, size=(chain * spec.layers, spec.depth,
                          spec.producers, rw), dtype=np.int32)  # fmt: skip
    x_host = rng.integers(-1000, 1000, size=(rw,), dtype=np.int32)
    ws = []
    for _ in range(copies):
        ws.append(dev.put(w_host))
    x = dev.put(x_host)
    return ToyBufs(spec, scratch, nbytes, step, h, ws, x, w_host, x_host, seed, extra)


def seed_words(spec: T.ToyChainSpec, seed: int) -> np.ndarray:
    """The first launch's ``h_in``: ``seed`` at :func:`toy_chain.seed_word`."""
    h = np.zeros(H_WORDS, np.int32)
    h[T.seed_word(spec)] = seed
    return h


def layer_bytes(spec: T.ToyChainSpec) -> int:
    return spec.layers * spec.depth * spec.producers * max(spec.row_words, 1) * 4


def launch_args(bufs: ToyBufs, k: int, layer: int, copy: int = 0) -> bytes:
    """Args of launch ``k`` of a chain (h[k] -> h[k+1], weights of launch k)."""
    s = bufs.spec
    w = bufs.w[copy] + k * layer_bytes(s)
    return pack_args(bufs.step, layer, scratch=bufs.scratch, h_in=bufs.h[k],
                     h_out=bufs.h[k + 1], w_down=w, w_up=bufs.x,
                     **(bufs.extra or {}))  # fmt: skip


def model_chain(bufs: ToyBufs, step: int, layers0: list[int], seed: int | None = None):
    """Host model of a chain of launches at ``step`` (layer id of launch k:
    ``layers0[k]``). Returns per launch the ``toy_reference`` dicts."""
    s = bufs.spec
    r = bufs.seed if seed is None else seed
    outs = []
    for k, l0 in enumerate(layers0):
        tags = [step * mbx.LAYER_SLOTS + l0 + li + 1 for li in range(s.layers)]
        w = bufs.w_host[k * s.layers:(k + 1) * s.layers]
        o = T.toy_reference(s, r, bufs.x_host, w, tags)
        outs.append(o)
        r = o["out"][-1]
    return outs


def checksum_model(bufs: ToyBufs, step0: int, steps: int, layers0: list[int]) -> dict:
    """Running ``h_out[4 li + 1]`` sums over ``steps`` steps, closed form:
    ``r_i = D_i + r_prev * P (P + 1) / 2 + P * tag`` (mod 2**32) with ``D_i``
    the level's summed dot products (step independent)."""
    s = bufs.spec
    P = s.producers
    S = P * (P + 1) // 2
    dsum = []
    for k in range(len(layers0)):
        per = []
        for li in range(s.layers):
            w = bufs.w_host[k * s.layers + li]
            if s.work:
                d = (w.astype(np.int64) * bufs.x_host.astype(np.int64)).sum(axis=2).sum(axis=1)
            else:
                d = np.zeros(s.depth, np.int64)
            per.append([int(v) for v in d])
        dsum.append(per)
    acc: dict = {}
    for st in range(step0, step0 + steps):
        r = bufs.seed
        for k, l0 in enumerate(layers0):
            for li in range(s.layers):
                tag = st * mbx.LAYER_SLOTS + l0 + li + 1
                for i in range(s.depth):
                    r = wrap32(dsum[k][li][i] + r * S + P * tag)
                acc[(k, li)] = wrap32(acc.get((k, li), 0) + r)
    return acc


def read_i32(dev, addr, n):
    return dev.get(addr, n * 4, np.int32)


def check_state(dev: H.Dev, bufs: ToyBufs, step: int, layers0: list[int],
                checksum_launches: int = 1) -> list[str]:  # fmt: skip
    """Compare every mailbox of the LAST launch of the chain, and every
    launch's h_out, with the model. Returns problems (empty = pass)."""
    s = bufs.spec
    lay = T.toy_layout(s)
    model = model_chain(bufs, step, layers0)
    errs = []
    last = model[-1]
    for li in range(s.layers):
        slot = (li & 1) if (s.layers > 1 and s.parity) else 0
        if any(((l2 & 1) if s.parity else 0) == slot for l2 in range(li + 1, s.layers)):
            continue  # a later layer of the launch reused the slot
        tag = step * mbx.LAYER_SLOTS + layers0[-1] + li + 1
        for i in range(s.depth):
            off = lay.offset(f"y{i}", 0, slot)
            raw = read_i32(dev, bufs.scratch + off, 2 * s.producers).reshape(-1, 2)
            if not (raw[:, 1].astype(np.int64) == wrap32(tag)).all():
                errs.append(f"layer {li} y{i}: tags != {tag}")
            if not (raw[:, 0] == last["y"][li][i]).all():
                bad = int(np.nonzero(raw[:, 0] != last["y"][li][i])[0][0])
                errs.append(f"layer {li} y{i}[{bad}] {raw[bad, 0]} != {last['y'][li][i][bad]}")
            if s.hop != "a2a":
                rr = read_i32(dev, bufs.scratch + lay.offset(f"r{i}", 0, slot), 2)
                if int(rr[1]) != wrap32(tag) or int(rr[0]) != wrap32(last["r"][li][i]):
                    errs.append(f"layer {li} r{i}: {rr.tolist()} != {[last['r'][li][i], tag]}")
            if s.hop == "last_arriver":
                c = read_i32(dev, bufs.scratch + lay.offset(f"c{i}", 0, slot), 1)
                if int(c[0]) != 0:
                    errs.append(f"layer {li} c{i}: counter {int(c[0])} not reset")
    for k, o in enumerate(model):
        hv = read_i32(dev, bufs.h[k + 1], H_WORDS)
        for li in range(s.layers):
            if int(hv[4 * li]) != wrap32(o["out"][li]):
                errs.append(f"launch {k} h_out[{4 * li}] {int(hv[4 * li])} != {o['out'][li]}")
    return errs


def check_launch(dev: H.Dev, bufs: ToyBufs, step: int, layers0: list[int], k: int) -> list[str]:
    """Every mailbox launch ``k`` of a chain left behind (the slots no later
    layer of that launch reused) against the host model, read right after
    that launch (search/INVENTORY.md merge item M10: :func:`check_state` reads
    only the LAST launch's mailboxes, and at depth >= 8 with P = 160 a fault
    confined to an earlier launch vanishes mod 2^32 from every later value,
    since each level multiplies ``r_prev`` by ``P (P + 1) / 2 = 2^4 x 805``).
    Pair with :func:`run_chain_checked`."""
    s = bufs.spec
    lay = T.toy_layout(s)
    o = model_chain(bufs, step, layers0)[k]
    errs = []
    for li in range(s.layers):
        slot = (li & 1) if (s.layers > 1 and s.parity) else 0
        if any(((l2 & 1) if s.parity else 0) == slot for l2 in range(li + 1, s.layers)):
            continue
        tag = step * mbx.LAYER_SLOTS + layers0[k] + li + 1
        for i in range(s.depth):
            raw = read_i32(dev, bufs.scratch + lay.offset(f"y{i}", 0, slot), 2 * s.producers).reshape(-1, 2)
            if not (raw[:, 1].astype(np.int64) == wrap32(tag)).all():
                errs.append(f"launch {k} layer {li} y{i}: tags != {tag}")
            if not (raw[:, 0] == o["y"][li][i]).all():
                bad = int(np.nonzero(raw[:, 0] != o["y"][li][i])[0][0])
                errs.append(f"launch {k} layer {li} y{i}[{bad}] {raw[bad, 0]} != {o['y'][li][i][bad]}")
    return errs


def run_chain_checked(dev, bufs, art, *, step: int, layers0: list[int]) -> list[str]:
    """:func:`run_chain_once`, one launch at a time, with :func:`check_launch`
    after every launch and :func:`check_state` at the end: no fault in any
    launch of the chain can hide (M10)."""
    fn = dev.function(art)
    dev.write(bufs.step, np.array([step], np.int32))
    errs = []
    for k, l0 in enumerate(layers0):
        dev.launch(fn, bufs.spec.grid, bufs.spec.threads, launch_args(bufs, k, l0))
        dev.sync()
        errs += check_launch(dev, bufs, step, layers0, k)
    return errs + check_state(dev, bufs, step, layers0)


def compile_art(spec_or_kernel, backend=None):
    k = spec_or_kernel if not isinstance(spec_or_kernel, T.ToyChainSpec) else T.build_toy_chain(spec_or_kernel)
    return H.compile_for(k, ARCH, backend)


def run_chain_once(dev, bufs, art, *, step: int, layers0: list[int]) -> None:
    fn = dev.function(art)
    dev.write(bufs.step, np.array([step], np.int32))
    for k, l0 in enumerate(layers0):
        dev.launch(fn, bufs.spec.grid, bufs.spec.threads, launch_args(bufs, k, l0))
    dev.sync()


# ------------------------------------------------------------- stamps


def decode_stamps(st: np.ndarray, names, table) -> dict | None:
    """``st``: ``[GRID][POINTS]`` int64 of one layer. The harness timeline
    decoder's fields (ticks): per stage ``end``/``incr``/``span``/``wait``/
    ``tail`` (medians over tasks), plus ``total``, ``entry_skew``, ``ramp``
    (first stage's first ready (or enter) from the first CTA entry)."""
    entries = [int(r[0]) for r in st if r[0]]
    if not entries:
        return None
    t0 = min(entries)
    out = {"stages": {}, "entry_skew": max(entries) - t0}
    prev_end = 0
    for i, n in enumerate(names):
        ctas = [table.cta_of(n, t) for t in range(min(table.stage(n).tasks, table.grid))]
        ent, ext, waits, tails = [], [], [], []
        for c in ctas:
            e, x = int(st[c][stp.enter_ix(i)]), int(st[c][stp.exit_ix(i)])
            if not e or not x:
                continue
            r = max(int(st[c][stp.ready_ix(i, w)]) for w in range(stp.WAVES))
            ent.append(e - t0)
            ext.append(x - t0)
            if r:
                waits.append(r - e)
                tails.append(x - r)
        if not ext:
            out["stages"][n] = None
            continue
        end = max(ext)
        out["stages"][n] = {
            "end": end, "incr": max(0, end - prev_end), "span": end - min(ent),
            "first_enter": min(ent), "wait_med": statistics.median(waits) if waits else 0,
            "tail_med": statistics.median(tails) if tails else 0,
        }  # fmt: skip
        prev_end = max(prev_end, end)
    lasts = [int(r[stp.exit_ix(len(names) - 1)]) for r in st if r[stp.exit_ix(len(names) - 1)]]
    out["total"] = (max(lasts) - t0) if lasts else prev_end
    s0 = out["stages"].get(names[0])
    out["ramp"] = s0["first_enter"] if s0 else None
    out["t0"] = t0
    out["t_end"] = t0 + out["total"]
    return out


def rebase_stamps(st: np.ndarray, t0: int) -> np.ndarray:
    """Copy of ``st`` with every stamped CTA's point 0 set to ``t0``.

    In a multi-layer launch, point 0 of layer ``li > 0`` is when each CTA
    finished the previous layer's stamp flush (idle CTAs: almost at the
    kernel entry), not a launch entry: decoding it as an entry gives a
    meaningless ramp, a total that starts at the earliest idle CTA and a
    negative launch boundary. Rebased to the previous layer's end, this
    decoder and the harness ``timeline.decode`` measure the layer from where
    the previous one ended (entry skew 0)."""
    out = st.copy()
    out[out[:, 0] != 0, 0] = t0
    return out


def decode_launch(sts, names, table) -> list[dict | None]:
    """Decode every layer of one launch (``sts[li]``: :func:`read_stamps` of
    layer id ``layer0 + li``): layer 0 from the kernel entry; later layers of
    a multi-layer launch rebased to the previous layer's end
    (:func:`rebase_stamps`), with ``ramp`` None (no launch ramp inside a
    launch). Each decode carries ``li`` (layer index inside the launch)."""
    out: list[dict | None] = []
    prev = None
    for li, st in enumerate(sts):
        if li == 0:
            d = decode_stamps(st, names, table)
        elif prev is None:
            d = None  # nothing to rebase on
        else:
            d = decode_stamps(rebase_stamps(st, prev["t_end"]), names, table)
            if d:
                d["ramp"] = None
        if d:
            d["li"] = li
        out.append(d)
        prev = d
    return out


def read_stamps(dev, bufs, layer: int) -> np.ndarray:
    lay = T.toy_layout(bufs.spec)
    G = bufs.spec.grid
    off = lay.stamp_base + layer * G * stp.POINTS * 8
    return dev.get(bufs.scratch + off, G * stp.POINTS * 8, np.int64).reshape(G, stp.POINTS)


# -------------------------------------------------------------- graphs


def graph_launches(dev, bufs, art, step_art, *, launches: int, copies: int,
                   stamps_layers: bool = True):  # fmt: skip
    """A function issuing ``launches`` chained toy launches (layer ids
    ``k * spec.layers``) then the step increment, for ``_Graph.capture``."""
    fn = dev.function(art)
    sfn = dev.function(step_art)
    s = bufs.spec

    def issue(stream: int) -> None:
        for k in range(launches):
            dev.launch(fn, s.grid, s.threads, launch_args(bufs, k, k * s.layers, k % copies),
                       stream=stream)  # fmt: skip
        dev.launch(sfn, 1, 64, struct.pack("<Q", bufs.step), stream=stream)

    return issue


def replay_check(dev, spec: T.ToyChainSpec, replays: int, *, chain: int = 2) -> dict:
    """``replays`` graph replays of a ``chain``-launch graph plus the step
    increment; checks the last state and the running checksums."""
    bufs = alloc(dev, spec, chain=chain)
    art, used = compile_art(spec)
    sart, _ = compile_art(T.build_step_kernel())
    # straddle the tag wrap at step 2**26 (the u32 tag repeats from there)
    step0 = (1 << 26) - replays // 2
    dev.write(bufs.step, np.array([step0], np.int32))
    g = H._Graph(dev)
    try:
        g.capture(graph_launches(dev, bufs, art, sart, launches=chain, copies=1))
        for _ in range(replays):
            g.lib.hipGraphLaunch(g.exec, g.stream)
        g._chk(g.lib.hipStreamSynchronize(g.stream), "sync")
    finally:
        g.close()
    layers0 = [k * spec.layers for k in range(chain)]
    last_step = step0 + replays - 1
    errs = check_state(dev, bufs, last_step, layers0)
    # running checksums: h[k+1][4 li + 1] = sum over steps of out
    want = checksum_model(bufs, step0, replays, layers0)
    for (k, li), v in want.items():
        got = int(read_i32(dev, bufs.h[k + 1], H_WORDS)[4 * li + 1])
        if got != v:
            errs.append(f"checksum launch {k} layer {li}: {got} != {v}")
    stepv = int(read_i32(dev, bufs.step, 1)[0])
    if stepv != wrap32(step0 + replays):
        errs.append(f"step counter {stepv} != {step0 + replays}")
    return {"name": spec.name(), "backend": used, "replays": replays, "chain": chain,
            "step0": step0, "ok": not errs, "errors": errs[:10]}  # fmt: skip


FLUSH_BYTES = 1 << 29  # > L2 + MALL: a memset of it evicts both


def time_cell(dev, spec: T.ToyChainSpec, *, launches: int = 12, copies: int = 1,
              reps: int = 5, art=None, step_art=None, bufs=None, flush: int = 0) -> dict:  # fmt: skip
    """Graph of ``launches`` chained launches (each its own layer weights)
    replayed ``reps`` times, each replay after a cache flush (``flush``: a
    device buffer of ``FLUSH_BYTES``, memset between replays; 0 = warm);
    per replay: event time per launch, and for stamped builds the decoded
    stamps of every launch."""
    s = spec
    if s.layers * launches > 62:
        raise ValueError("layer ids of a timed graph must stay < 63")
    bufs = bufs or alloc(dev, s, chain=launches, copies=copies)
    art = art or compile_art(s)[0]
    step_art = step_art or compile_art(T.build_step_kernel())[0]
    dev.write(bufs.step, np.array([1000], np.int32))
    g = H._Graph(dev)
    rec = {"name": s.name(), "launches": launches, "copies": copies, "per_launch_ms": [],
           "stamps": []}  # fmt: skip
    table = T.toy_table(s)
    names = table.order
    try:
        g.capture(graph_launches(dev, bufs, art, step_art, launches=launches, copies=copies))
        g.replay()  # warm (code objects, TLB)
        for _ in range(reps):
            if flush:
                dev.rt.memset(flush, 0, FLUSH_BYTES)
                dev.sync()
            e0, e1 = dev.rt.event(), dev.rt.event()
            e0.record(g.stream.value)
            g.lib.hipGraphLaunch(g.exec, g.stream)
            e1.record(g.stream.value)
            e1.synchronize()
            rec["per_launch_ms"].append(e0.elapsed_to(e1) / launches)
            e0.destroy()
            e1.destroy()
            if s.knobs.stamps:
                per = []
                for k in range(launches):
                    sts = [read_stamps(dev, bufs, k * s.layers + li) for li in range(s.layers)]
                    per += decode_launch(sts, names, table)
                rec["stamps"].append(per)
    finally:
        g.close()
    step = int(read_i32(dev, bufs.step, 1)[0]) - 1
    errs = check_state(dev, bufs, step, [k * s.layers for k in range(launches)])
    rec["ok"] = not errs
    rec["errors"] = errs[:5]
    return rec


def summarize(rec: dict) -> dict:
    """Medians (ticks) of a stamped time_cell record: total per layer, ramp,
    entry skew, per-stage incr, launch boundary gap (next launch's first
    entry minus this launch's last exit, in-graph). Multi-layer launches
    (:func:`decode_launch`): ramps and entry skews of launch entries only
    (layer index 0), boundaries only into a launch's first layer; later
    layers' totals run from the previous layer's end."""
    if not rec["stamps"]:
        return {}
    tot, ramp, skew, gaps, incr = [], [], [], [], {}
    for per in rec["stamps"]:
        ds = [d for d in per if d]
        for d in ds:
            tot.append(d["total"])
            if d.get("li", 0) == 0:
                ramp.append(d["ramp"])
                skew.append(d["entry_skew"])
            for n, st in d["stages"].items():
                if st:
                    incr.setdefault(n, []).append(st["incr"])
        for a, b in zip(ds, ds[1:]):
            if b.get("li", 0) == 0:
                gaps.append(b["t0"] - a["t_end"])
    med = statistics.median
    return {"total": med(tot), "ramp": med(ramp), "entry_skew": med(skew),
            "boundary": med(gaps) if gaps else None,
            "incr": {n: med(v) for n, v in incr.items()}}  # fmt: skip


# ----------------------------------------------------------------- CLI


def _spec(depth, work, hop, consume, layers=1, stamps=True, **kw) -> T.ToyChainSpec:
    knobs = FrameworkKnobs(stamps=stamps, **kw.pop("fk", {}))
    return T.ToyChainSpec(depth=depth, work=work, hop=hop, consume=consume,
                          layers=layers, knobs=knobs, **kw)  # fmt: skip


def check_grid() -> list[T.ToyChainSpec]:
    out = []
    for hop in T.HOPS:
        for consume in T.CONSUMES:
            for work in (0, 8):
                for layers in (1, 2):
                    out.append(_spec(3, work, hop, consume, layers, stamps=False))
    out.append(_spec(3, 8, "a2a", "wave0", stamps=True))
    out.append(_spec(3, 8, "a2a", "wave0", prefetch="l2_touch", stamps=False))
    out.append(_spec(3, 8, "a2a", "wave0", layers=2, parity=False, stamps=False))
    for pl in ("sequential", "xcd_blocked"):
        out.append(_spec(3, 0, "a2a", "wave0", placement=pl, stamps=False))
    out.append(_spec(3, 0, "a2a", "wave0", stamps=False,
                     fk={"mailbox": mbx.MailboxKnobs(numa_copies=1, pause="s_sleep1")}))
    for cal in ("", "s_down"):
        for hop in ("a2a", "last_arriver"):
            out.append(_spec(3, 8, hop, "wave0", stamps=False,
                             fk={"mailbox": mbx.MailboxKnobs(numa_copies=2),
                                 "numa_flip_arg": cal}))  # fmt: skip
    for hop in T.HOPS:  # the Python-engine-only mailbox (tagged_poll)
        out.append(_spec(3, 8, hop, "wave0", stamps=False,
                         fk={"mailbox": mbx.MailboxKnobs(impl="ir")}))
    out.append(_spec(3, 0, "reduce_task", "waves", layers=2, stamps=False,
                     fk={"mailbox": mbx.MailboxKnobs(numa_copies=2), "numa_flip_arg": "s_down"}))
    for sf in (False, True):  # g0 weights at the kernel entry (+ step load first)
        for pl in ("atom", "xcd_blocked"):
            out.append(_spec(3, 8, "a2a", "wave0", early="entry", placement=pl, stamps=False,
                             fk={"step_first": sf}))  # fmt: skip
    out.append(_spec(3, 8, "last_arriver", "wave0", layers=2, early="entry", stamps=False,
                     fk={"step_first": True}))
    for early in ("off", "entry"):  # scalar step-counter load
        out.append(_spec(3, 8, "a2a", "wave0", early=early, reduce="dpp", stamps=False,
                         fk={"step_load": "smem", "mailbox": mbx.MailboxKnobs(numa_copies=2),
                             "numa_flip_arg": "s_down"}))  # fmt: skip
    out.append(_spec(3, 0, "a2a", "wave0", layers=2, stamps=True, fk={"step_load": "smem"}))
    # idle-CTA prefetch in 2-layer launches (L2-hot next-level weights: the
    # LDS-parity regression of the review-fix round), every hop
    for hop, consume in (("a2a", "waves"), ("a2a", "wave0"), ("reduce_task", "wave0"),
                         ("last_arriver", "wave0")):
        out.append(_spec(4, 8, hop, consume, layers=2, prefetch="l2_touch", reduce="dpp",
                         stamps=False, fk={"mailbox": mbx.MailboxKnobs(numa_copies=2),
                                           "numa_flip_arg": "s_down", "kernarg_preload": 8}))
    for kp in (3, 8):  # kernarg preload (Python engine)
        out.append(_spec(3, 8, "a2a", "wave0", reduce="dpp", stamps=False,
                         fk={"kernarg_preload": kp, "mailbox": mbx.MailboxKnobs(numa_copies=2),
                             "numa_flip_arg": "s_down"}))  # fmt: skip
    out.append(_spec(3, 0, "last_arriver", "wave0", layers=2, stamps=True, fk={"kernarg_preload": 8}))
    return out


def cmd_check(dev, args) -> int:
    bad = 0
    for spec in check_grid():
        bufs = alloc(dev, spec, chain=2)
        art, used = compile_art(spec)
        ok_all = True
        for step in (5, (1 << 26) - 1):
            run_chain_once(dev, bufs, art, step=step, layers0=[0, spec.layers])
            errs = check_state(dev, bufs, step, [0, spec.layers])
            ok_all &= not errs
            if errs:
                print(spec.name(), "step", step, "FAIL", errs[:3], flush=True)
        print(f"{spec.name():<60} backend={used} {'ok' if ok_all else 'FAIL'}", flush=True)
        bad += not ok_all
    return 1 if bad else 0


def cmd_replay(dev, args) -> int:
    bad = 0
    for spec in (_spec(4, 8, "a2a", "wave0", stamps=False),
                 _spec(4, 0, "last_arriver", "wave0", stamps=False),
                 _spec(3, 8, "a2a", "wave0", layers=2, stamps=False)):  # fmt: skip
        r = replay_check(dev, spec, args.replays)
        print(json.dumps(r), flush=True)
        bad += not r["ok"]
    return 1 if bad else 0


def cmd_time(dev, args) -> int:
    """Depth sweep and lever arms, interleaved; ratios vs the a2a/wave0 arm
    at the same depth."""
    arms = {}
    for d in args.depths:
        for work in args.works:
            arms[("a2a", d, work)] = _spec(d, work, "a2a", "wave0")
            arms[("a2a_waves", d, work)] = _spec(d, work, "a2a", "waves")
            arms[("reduce", d, work)] = _spec(d, work, "reduce_task", "wave0")
            arms[("last", d, work)] = _spec(d, work, "last_arriver", "wave0")
    cells = {}
    for key, spec in arms.items():
        art, used = compile_art(spec)
        cells[key] = (spec, art, alloc(dev, spec, chain=args.launches))
    sart = compile_art(T.build_step_kernel())[0]
    flush = dev.alloc(FLUSH_BYTES, zero=False)
    raw = {str(k): [] for k in arms}
    keys = list(arms)
    for rnd in range(args.rounds):
        for key in keys if rnd % 2 == 0 else keys[::-1]:
            spec, art, bufs = cells[key]
            rec = time_cell(dev, spec, launches=args.launches, reps=args.reps, art=art,
                            step_art=sart, bufs=bufs, flush=flush)  # fmt: skip
            raw[str(key)].append({"per_launch_ms": rec["per_launch_ms"], "ok": rec["ok"],
                                  "summary": summarize(rec), "errors": rec["errors"]})  # fmt: skip
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "toy_time.json"), "w") as fh:
            json.dump(raw, fh, indent=1, default=str)
    ok = all(r["ok"] for v in raw.values() for r in v)
    for key in keys:
        tot = statistics.median(r["summary"]["total"] for r in raw[str(key)])
        ref = statistics.median(r["summary"]["total"] for r in raw[str(("a2a",) + key[1:])])
        print(f"{str(key):<28} total_vs_a2a={tot / ref:.3f}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("check", "replay", "time"))
    ap.add_argument("--replays", type=int, default=10000)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--works", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--launches", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    prob = H.env_problem()
    if prob:
        print("SKIP:", prob)
        return 2
    dev = H.Dev(watchdog_s=120)
    try:
        return {"check": cmd_check, "replay": cmd_replay, "time": cmd_time}[args.cmd](dev, args)
    finally:
        dev.close()


if __name__ == "__main__":
    sys.exit(main())

