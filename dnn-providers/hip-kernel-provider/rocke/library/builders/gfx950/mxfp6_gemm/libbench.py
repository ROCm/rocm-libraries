# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Library harness for the gfx950 MXFP6 GEMM tile-config library: plan, compile, verify, time, tabulate.

A *manifest* (JSON) names the configs (a library tile + kernel family + spec overrides + direct-compiler levers per
config), the element formats and the problem sets (``libtiles.problems``). The harness expands it into *cells*
(config x format x problem) and *builds* (one code object per cell, plus the cell's must-fail mutants and its
MFMA-only ceiling probe), then runs in resumable, sharded stages that share one run directory:

* ``plan``    -- expand the manifest, refuse configs the kernel rejects (``validate_spec``) and levers the direct
  compiler does not list (``target_compile --list-levers``), write ``plan.json``;
* ``compile`` -- one shard of the builds, each in its own fresh process (``bench.py compile``: no build inherits
  another's LLVM state), on CPU nodes (any number of shards in parallel);
* ``gpu``     -- one shard of the cells on one GPU: verify every cell (integer set bit-exact, Gaussian / uniform sets
  within the scaled-MFMA tolerance on distinct per-entry data, every launch deterministic, every mutant must fail),
  then time it against an A-vs-A twin (second module load of the same code object) and its ceiling probe:
  cold-start (post-idle) rounds and warm (steady-state) windows, arms shuffled per round, sysfs SCLK / power ->
  mJ per launch and pJ per FLOP, paired bootstrap CIs;
* ``table``   -- merge the cell results with the per-config ceilings (MFMA-only ceiling table + the bandwidth
  roofline) into a table: TF/s, % of ceiling, % of the product spec (``spec_tflops`` of the ceilings input), twin
  noise, verification;
* ``status``  -- the ledger (every stage appends JSON lines to ``ledger/<stage>-<shard>-<job>.jsonl``; a stage skips
  the work that a ledger already records as done, so every stage is resumable).

Python API: :func:`load_manifest`, :func:`plan`, :func:`compile_shard`, :func:`gpu_shard`, :func:`table`,
:func:`status`. ``MXFP6_LIBBENCH_FAKE=1`` runs every stage without a compiler or a GPU (pipeline self-test).

Data / timing protocol (see ``notes/tile-library.md`` "Benchmark problems"): verification runs the full-size
launch with ``n_src`` distinct data sources cycled over the batch entries (entry e holds source e mod n_src) and
checks every entry against its source's float64 reference; timing uses Gaussian (+ uniform) data in ``rotate``
distinct buffer sets cycled launch by launch (input footprint >= 2 x MALL).

Absolute figures are for the operator's internal record only; repository documents carry ratios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from builders.gfx950.mxfp6_gemm import libtiles as T

MUTANTS = ("scale_sel", "k_order", "drop_kstep", "wrong_fmt")
#: A-vs-A twin gate: |median twin ratio - 1| and half the CI width must stay within TWIN_TOL, else the warm windows
#: are extended (up to TWIN_RETRIES more sets); a cell that still fails is flagged (``twin_gate`` false)
TWIN_TOL = 0.02
TWIN_RETRIES = 2
#: spec fields every library cell starts from (below the manifest defaults): problem-shape-independent CTA order
LIBRARY_DEFAULTS: Dict[str, object] = {"cta_order": "grouped", "xcd_block": "none"}
DEFAULT_LEVERS = "scalesep,dsrebase,dshoist,m0fill,m0cse"


def _fake() -> bool:
    return os.environ.get("MXFP6_LIBBENCH_FAKE") == "1"


def _sha(obj, n: int = 16) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def _median(v: Sequence[float]) -> Optional[float]:
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def _kmod():
    from kernels.gfx950.mxfp6_gemm import gemm

    return gemm


# ================================================================== manifest and plan


@dataclass
class Config:
    """One library entry: a tile, a kernel family and the spec overrides that realize it."""

    id: str
    tile: str  # "MT0xMT1xDepthU"
    family: str = "ak"  # libtiles.FAMILIES
    spec: Dict[str, object] = field(default_factory=dict)  # Mxfp6GemmSpec overrides (geometry, schedule, levers...)
    levers: Optional[str] = None  # direct-compiler levers (None: the manifest default)
    compiler: str = "rust"  # bench.py compile --compiler: rust | llvm | auto
    #: spec base: "family" (library.family_spec, the first-build default), "config" (library.config_spec, the kernel
    #: the library ships per config) or "dispatch" (library.dispatch: config_spec + the per-problem schedule, i.e. what
    #: a user gets); ``spec`` is applied on top of the base
    base: str = "family"
    #: explicit problems of this config only ({set, batch, m, n, k[, rotate]}); replaces the manifest sets / problems
    problems: Optional[List[Dict[str, object]]] = None
    #: am lane (default ""): "transpose" runs an ``am`` config as the library's transpose path -- the M-contiguous A
    #: through the transpose kernel (``transpose.pick`` + ``library.AM_TRANSPOSE`` + ``tr`` overrides) into a
    #: workspace, then the config's ``ak`` kernel (spec base as above); "plan": ``library.am_plan`` decides per problem
    via: str = ""
    tr: Dict[str, object] = field(default_factory=dict)


@dataclass
class Manifest:
    name: str
    configs: List[Config]
    formats: Tuple[str, ...] = ("e2m3",)
    sets: Tuple[str, ...] = ("S",)  # libtiles problem sets: S, G, S-k4096 / S-k8192 / S-k16384, R (all R problems)
    problems: Tuple[Dict[str, object], ...] = ()  # explicit problems {batch, m, n, k[, rotate, set]} for every config
    defaults: Dict[str, object] = field(default_factory=dict)  # spec overrides applied to every config first
    levers: Optional[str] = None  # default: the kernel's library.LEVERS (family builds), else DEFAULT_LEVERS
    compiler_path: Optional[str] = None  # target_compile (default: $ROCKE_RUST_COMPILE)
    verify_dists: Tuple[str, ...] = ("int", "gauss", "uniform")
    time_dists: Tuple[str, ...] = ("gauss",)
    mutants: bool = True
    ceiling_probe: bool = True  # time the config's own probe="mfma_only" build next to it (when it builds)
    n_src: int = 3  # distinct data sources cycled over the batch entries in verification
    #: per GPU worker process: run a build's must-fail mutants once (on its first cell) and reuse the verdict for its
    #: other problems; time its MFMA-only probe arm once and reuse the ceiling (problem suites: one build, many cells)
    mutants_once: bool = False
    probe_once: bool = False
    cold_rounds: int = 15
    warm_windows: int = 3
    warm_s: float = 2.0
    gap_ms: float = 25.0
    seed: int = 42


def load_manifest(path_or_dict) -> Manifest:
    d = path_or_dict if isinstance(path_or_dict, Mapping) else json.loads(Path(path_or_dict).read_text())
    d = dict(d)
    cfgs = [Config(**c) for c in d.pop("configs")]
    for k in ("formats", "sets", "verify_dists", "time_dists", "problems"):
        if k in d:
            d[k] = tuple(d[k])
    return Manifest(configs=cfgs, **d)


def family_spec(family: str) -> Dict[str, object]:
    """Spec fields of a kernel family: ``am`` reads the activation (our A, hipBLASLt B) M-contiguous
    (``a_layout="mcontig"``)."""
    if family == "ak":
        return {}
    if family == "am":
        return {"a_layout": "mcontig"}
    raise ValueError(f"unknown family {family!r} (one of {sorted(T.FAMILIES)})")


def tile_spec(tile: T.Tile, lay: Optional[T.Layout] = None) -> Dict[str, object]:
    lay = lay or T.default_layout(tile)
    return {"tile_m": tile.tile_m, "tile_n": tile.tile_n, "depth_u": tile.du, "atom": lay.atom, "waves_m": lay.waves_m,
            "waves_n": lay.waves_n}


def problems_of(tile: T.Tile, sets: Sequence[str], explicit: Sequence[Mapping[str, object]] = ()) -> List[T.Problem]:
    out = []
    for e in explicit:
        p = T.Problem(str(e.get("set", "X")), tile.name, int(e["batch"]), int(e["m"]), int(e["n"]), int(e["k"]))
        p.tiles = p.batch * (p.m // tile.tile_m) * (p.n // tile.tile_n)
        p.rotate = int(e["rotate"]) if "rotate" in e else max(1, -(-2 * T.MALL_BYTES // int(p.in_bytes)))
        out.append(p)
    for s in sets:
        if s in ("S", "G"):
            out += [p for p in T.problems(tile) if p.set == s]
        elif s.startswith("S-k"):
            out += [p for p in T.k_sweep(tile) if p.set == s]
        elif s == "R":
            out += [p for p in T.realistic() if p.m % tile.tile_m == 0 and p.n % tile.tile_n == 0 and p.k % tile.du == 0]
        else:
            raise ValueError(f"unknown problem set {s!r}")
    return out


def problem_spec(p: T.Problem) -> Dict[str, object]:
    return {"m": p.m, "n": p.n, "k": p.k, "batch": p.batch}


@dataclass
class Cell:
    """One measurement: a config at one element format and one problem."""

    id: str
    config: str
    tile: str
    family: str
    fmt: str
    problem: Dict[str, object]
    spec: Dict[str, object]  # full override dict of the candidate build
    build: str  # build id of the candidate
    mutants: Dict[str, str] = field(default_factory=dict)  # mutant -> build id
    probe: Optional[str] = None  # build id of the probe="mfma_only" build (None: not buildable)
    probe_note: str = ""
    levers: str = ""
    compiler: str = "rust"
    #: am lane: the transpose kernel run before the GEMM (``transpose.TrSpec`` fields; None: the GEMM alone)
    pre: Optional[Dict[str, object]] = None


def _validate(over: Mapping[str, object], prob: Optional[T.Problem] = None) -> Tuple[bool, str]:
    """``validate_spec``, plus ``runtime_problem_ok`` for a runtime-shape build and its problem."""
    if _fake():
        return True, ""
    g = _kmod()
    try:
        spec = replace(g.Mxfp6GemmSpec(), **dict(over))
    except TypeError as e:  # a field the kernel does not have (yet)
        return False, f"spec field: {e}"
    ok, why = g.validate_spec(spec)
    if ok and prob is not None and getattr(spec, "runtime_shape", False) and hasattr(g, "runtime_problem_ok"):
        ok, why = g.runtime_problem_ok(spec, prob.m, prob.n, prob.k, prob.batch)
    return ok, why


def _library_module():
    try:
        from kernels.gfx950.mxfp6_gemm import library

        return library if hasattr(library, "family_spec") else None
    except ImportError:
        return None


def cell_spec(tile: T.Tile, family: str, fmt: str, prob: T.Problem, over: Mapping[str, object],
              base: str = "family") -> Dict[str, object]:
    """The full spec override dict of one cell.

    With the kernel's ``library.family_spec`` (the kernel agent's first-build default per config and family): that
    spec (``over`` applied inside it, K reference in the problem's residue class); a static-shape result takes the
    problem's m / n / k / batch, a runtime-shape build keeps its reference shape (the problem is a launch argument,
    so one build serves every problem of its K residue). Without it: ``LIBRARY_DEFAULTS`` < ``tile_spec`` <
    ``family_spec`` (this module) < ``over`` < the problem shape."""
    lib = None if _fake() else _library_module()
    if lib is not None:
        if base == "dispatch":
            sp = lib.dispatch(tile.mt0, tile.mt1, tile.du, family, fmt, prob.m, prob.n, prob.batch, k_target=prob.k,
                              **dict(over))
        elif base == "config":
            sp = lib.config_spec(tile.mt0, tile.mt1, tile.du, family, fmt, k_target=prob.k, **dict(over))
        elif base == "family":
            sp = lib.family_spec(tile.mt0, tile.mt1, tile.du, family, fmt, k_target=prob.k, **dict(over))
        elif base == "select":  # the split-K lane's per-problem choice on top of dispatch (splitk.select)
            from kernels.gfx950.mxfp6_gemm import splitk as SK

            sp = SK.select_config(tile.mt0, tile.mt1, tile.du, family, fmt, prob.m, prob.n, prob.k, prob.batch,
                                  **dict(over))
        else:
            raise ValueError(f"unknown spec base {base!r} (family | config | dispatch | select)")
        d = asdict(sp)
        if not d.get("runtime_shape"):
            d.update(problem_spec(prob))
        return d
    return {**LIBRARY_DEFAULTS, **tile_spec(tile), **family_spec(family), **dict(over), **problem_spec(prob),
            "elem_fmt": fmt}


def _via(tile: T.Tile, c: "Config", fmt: str, p: T.Problem) -> Tuple[str, Optional[Dict[str, object]]]:
    """am lane: (kernel family, transpose spec dict or None) of an ``am`` config run ``via`` "transpose" (always) or
    "plan" (``library.am_plan``'s choice for the problem)."""
    if c.family != "am" or c.via not in ("transpose", "plan"):
        raise ValueError(f"via {c.via!r}: an am config, via transpose | plan")
    from kernels.gfx950.mxfp6_gemm import library as LB, transpose as TR

    if c.via == "plan" and LB.am_plan(tile.mt0, tile.mt1, tile.du, fmt, p.m, p.n, p.k, p.batch).path == "native":
        return "am", None
    ts = replace(TR.pick(p.m, p.k, p.batch), **{**LB.AM_TRANSPOSE, **dict(c.tr)})
    ok, why = TR.validate(ts)
    if not ok or p.m < 1 or p.k % ts.tk:  # any M (the M-padded input, transpose.padded_m)
        raise ValueError(f"transpose: {why or 'K a multiple of tk'}")
    return "ak", asdict(ts)


def default_levers() -> str:
    """Direct-compiler levers of the library family builds (``library.LEVERS``; e.g. ``trtag`` for the ``am``
    transposing reads), else :data:`DEFAULT_LEVERS`."""
    lib = None if _fake() else _library_module()
    return getattr(lib, "LEVERS", DEFAULT_LEVERS) if lib is not None else DEFAULT_LEVERS


def _wave_rows(over: Mapping[str, object]) -> Tuple[int, int]:
    """(rows per wave along M, rows per wave along N) of a spec override dict."""
    return (int(over.get("tile_m", 256)) // int(over.get("waves_m", 2)),
            int(over.get("tile_n", 256)) // int(over.get("waves_n", 2)))


#: per-process reuse of a build's mutant verdicts and MFMA-only probe ceiling (``mutants_once`` / ``probe_once``)
MEMO: Dict[str, Dict[str, object]] = {"mut": {}, "probe": {}}

#: the arm the GPU worker last launched (fault attribution in the ledger)
CURRENT_ARM: List[Optional[str]] = [None]


def _lib_mutants(family: str) -> List[str]:
    """The kernel's library mutants (``LIB_MUTANTS``: batch stride / mcontig swizzle) that apply to ``family``."""
    if _fake():
        return []
    mus = list(getattr(_kmod(), "LIB_MUTANTS", ()))
    sk = _kmod().__dict__.get("SK_MUTANTS", ())  # split-K fix-up must-fails (validated per spec: split builds only)
    from kernels.gfx950.mxfp6_gemm import splitk as _SK

    mus += [m for m in sk if m not in getattr(_SK, "SK_SHOULD_FAIL", ())]
    return [m for m in mus if not (m.startswith("mc_") and family != "am")]


def _probe_over(over: Mapping[str, object]) -> Tuple[Optional[Dict[str, object]], str]:
    """The config's MFMA-only probe build (tune.py's rule: mfma_only may need scale_src=vgpr)."""
    p = {**over, "probe": "mfma_only"}
    if p.get("splitk", "none") != "none":  # a split build's ceiling arm is its tile's data-parallel probe
        p.update(splitk="none", m_mask=False, sk_split=1, sk_hybrid=False)
    ok, why = _validate(p)
    if ok:
        return p, ""
    p2 = {**p, "scale_src": "vgpr"}
    ok2, why2 = _validate(p2)
    if ok2:
        return p2, "scale_src=vgpr (mfma_only needs it)"
    return None, f"probe not buildable: {why} / {why2}"


def compiler_levers(tool: Optional[str]) -> Optional[frozenset]:
    """``target_compile --list-levers`` (None: the tool predates the list, or there is no tool)."""
    if _fake():
        return frozenset(DEFAULT_LEVERS.split(",") + ["fakelever"])
    if not tool or not os.path.exists(tool):
        return None
    r = subprocess.run([tool, "--list-levers"], capture_output=True, text=True, timeout=60)
    names = r.stdout.split() if r.returncode == 0 else []
    return frozenset(names) if names else None


def unknown_levers(known: Optional[frozenset], levers: str) -> Optional[List[str]]:
    """Lever names ``known`` does not list (None: the compiler cannot list its levers: refuse)."""
    names = [n.strip().lstrip("-") for n in levers.replace("+", ",").split(",") if n.strip()]
    if known is None:
        return None
    return [n for n in names if n not in known]


def plan(man: Manifest, run: Path) -> dict:
    """Expand ``man`` into cells and builds under ``run`` (``plan.json``); refused configs are listed with the
    reason (kernel validation, unknown direct-compiler levers, a compiler that cannot list its levers)."""
    run.mkdir(parents=True, exist_ok=True)
    tool = man.compiler_path or os.environ.get("ROCKE_RUST_COMPILE")
    known = compiler_levers(tool)
    tool_sha = (hashlib.sha256(Path(tool).read_bytes()).hexdigest()[:16]
                if tool and os.path.exists(tool) and not _fake() else ("fake" if _fake() else None))
    builds: Dict[str, dict] = {}
    cells: List[Cell] = []
    refused: List[dict] = []

    def add_build(over: Mapping[str, object], levers: str, compiler: str, role: str) -> str:
        key = {"spec": dict(over), "levers": levers if compiler != "llvm" else "", "compiler": compiler,
               "tool": tool_sha if compiler != "llvm" else None}
        bid = _sha(key)
        builds.setdefault(bid, {**key, "id": bid, "role": role})
        return bid

    for c in man.configs:
        tile = T.Tile.parse(c.tile)
        levers = c.levers if c.levers is not None else (man.levers if man.levers is not None else default_levers())
        if c.compiler in ("rust", "auto"):
            unk = unknown_levers(known, levers)
            if unk is None:
                refused.append({"config": c.id, "why": f"direct compiler {tool} cannot list its levers (--list-levers)"})
                continue
            if unk:
                refused.append({"config": c.id, "why": f"unknown direct-compiler levers {unk} for {tool}"})
                continue
        for fmt in man.formats:
            for p in (problems_of(tile, (), c.problems) if c.problems is not None
                      else problems_of(tile, man.sets, man.problems)):
                cid = f"{c.id}|{fmt}|{p.name}"
                pre = None
                try:
                    fam = c.family
                    if c.via:  # am lane: the transpose path (or the library's per-problem plan)
                        fam, pre = _via(tile, c, fmt, p)
                    over = cell_spec(tile, fam, fmt, p, {**man.defaults, **c.spec}, c.base)
                except ValueError as e:  # the kernel's family_spec found no fit
                    refused.append({"config": c.id, "cell": cid, "why": f"family_spec: {e}"})
                    continue
                ok, why = _validate(over, p)
                if not ok:
                    refused.append({"config": c.id, "cell": cid, "why": f"validate: {why}"})
                    continue
                cell = Cell(cid, c.id, c.tile, c.family, fmt, asdict(p), over, add_build(over, levers, c.compiler, "cand"),
                            levers=levers, compiler=c.compiler, pre=pre)
                if man.mutants:
                    for mu in list(MUTANTS) + _lib_mutants("ak" if pre else c.family):
                        if mu == "scale_sel" and _wave_rows(over) == (16, 16):
                            continue  # one scale byte per wave side: op_sel has nothing to mis-select (scale_lane)
                        mo = {**over, "mutant": mu}
                        if _validate(mo, p)[0]:
                            cell.mutants[mu] = add_build(mo, levers, c.compiler, "mutant")
                if man.ceiling_probe:
                    po, note = _probe_over(over)
                    cell.probe_note = note
                    if po is not None:
                        cell.probe = add_build(po, levers, c.compiler, "probe")
                cells.append(cell)
    out = {"manifest": {**asdict(man), "configs": [asdict(c) for c in man.configs]}, "compiler": tool,
           "compiler_sha": tool_sha, "known_levers": sorted(known) if known else None,
           "cells": [asdict(c) for c in cells], "builds": builds, "refused": refused, "created": time.time()}
    (run / "plan.json").write_text(json.dumps(out, indent=1, default=str))
    (run / "ledger").mkdir(exist_ok=True)
    (run / "objs").mkdir(exist_ok=True)
    (run / "cells").mkdir(exist_ok=True)
    return out


# ================================================================== ledger


class Ledger:
    """Append-only JSON lines per (stage, shard, job); ``done`` merges every ledger file of the run."""

    def __init__(self, run: Path, stage: str, shard: str):
        job = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
        self.path = run / "ledger" / f"{stage}-{shard}-{job}-{os.getpid()}.jsonl"  # one writer per file
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, **kw) -> None:
        self.add_many([kw])

    def add_many(self, rows: Sequence[dict]) -> None:
        """Append several events in one write (a build batch: one NFS append, not one per build)."""
        host = os.uname().nodename if hasattr(os, "uname") else "local"
        lines = []
        for kw in rows:
            kw = dict(kw)
            kw.setdefault("t", time.time())
            kw.setdefault("host", host)
            lines.append(json.dumps(kw, default=str) + "\n")
        with open(self.path, "a") as f:
            f.write("".join(lines))

    @staticmethod
    def events(run: Path) -> List[dict]:
        out = []
        for p in sorted((run / "ledger").glob("*.jsonl")):
            for line in p.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut by a killed job
        return out

    @staticmethod
    def state(run: Path) -> Dict[str, str]:
        """Last event per item ("build:<id>" / "cell:<id>") -> its status."""
        st = {}
        for e in sorted(Ledger.events(run), key=lambda e: e.get("t", 0)):
            st[e["item"]] = e["status"]
        return st


def _shard(items: Sequence[str], shard: int, nshards: int) -> List[str]:
    return [x for i, x in enumerate(sorted(items)) if i % nshards == shard]


# ================================================================== compile stage


def compile_shard(run: Path, shard: int = 0, nshards: int = 1, jobs: int = 8, redo_failed: bool = False,
                  timeout_s: float = 900.0, cs: Optional["CompileServer"] = None) -> dict:
    """Compile this shard's builds: direct-compiler builds through a ``CompileServer`` (``cs``: a caller's
    long-lived one, e.g. the build service's; else one for this call), LLVM builds as fresh ``bench.py compile``
    processes (``jobs`` at once)."""
    pl = json.loads((run / "plan.json").read_text())
    led = Ledger(run, "compile", f"{shard}of{nshards}")
    st = Ledger.state(run)
    todo = []
    for bid in _shard(list(pl["builds"]), shard, nshards):
        s = st.get(f"build:{bid}")
        if s == "ok" and obj_path(run, bid) is not None:
            continue
        if s in ("failed", "spills") and not redo_failed:
            continue
        todo.append(bid)
    print(f"[compile] shard {shard}/{nshards}: {len(todo)} builds to do", flush=True)
    procs: Dict[str, tuple] = {}
    done = {"ok": 0, "failed": 0}
    tool = pl.get("compiler")
    if not _fake() and tool and os.path.exists(tool) and os.environ.get("LIBBENCH_COMPILE_SERVER", "1") == "1":
        # long-lived path: every direct-compiler build of the shard in this process, all lever sets in one batch
        rust = [bid for bid in todo if pl["builds"][bid]["compiler"] == "rust"]
        rs = set(rust)
        todo = [bid for bid in todo if bid not in rs]
        if cs is not None and cs.tool == tool:
            build_rust(run, pl, rust, cs, led, done)
            print(f"[compile] phases {cs.last_stats}", flush=True)
        else:
            with CompileServer(tool, Path(os.environ.get("LIBBENCH_OBJ_CACHE", str(run / "objcache"))),
                               jobs=jobs, pool=True) as cs_:
                build_rust(run, pl, rust, cs_, led, done)
                print(f"[compile] phases {cs_.last_stats}", flush=True)
    while todo or procs:
        while todo and len(procs) < jobs:
            bid = todo.pop(0)
            b = pl["builds"][bid]
            out = run / "objs" / f"{bid}.hsaco"
            if _fake():
                out.write_bytes(b"FAKE" + bid.encode())
                Path(str(out) + ".json").write_text(json.dumps({"ok": True, "name": "fake_" + bid, "fake": True,
                                                                "notes": {"vgpr_count": 128, "agpr_count": 256,
                                                                          "group_segment_fixed_size": 100000}}))
                led.add(item=f"build:{bid}", status="ok", role=b["role"], s=0.0)
                done["ok"] += 1
                continue
            env = dict(os.environ)
            if b["compiler"] != "llvm":
                env["ROCKE_RUST_LEVERS"] = b["levers"]
                if tool:
                    env["ROCKE_RUST_COMPILE"] = tool
            cmd = [sys.executable, "-m", "benchmarks.gfx950.mxfp6_gemm.bench", "compile", "--spec",
                   json.dumps(b["spec"]), "--out", str(out), "--compiler", b["compiler"]]
            procs[bid] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env),
                          time.time(), out)
        for bid in list(procs):
            p, t0, out = procs[bid]
            if p.poll() is None and time.time() - t0 < timeout_s:
                continue
            if p.poll() is None:
                p.kill()
            so, se = p.communicate()
            meta = {}
            mp = Path(str(out) + ".json")
            if p.returncode == 0 and out.exists() and mp.exists():
                meta = json.loads(mp.read_text())
            status = "ok" if meta.get("ok") else "failed"
            if status == "ok" and _spills(meta):
                status = "spills"
            led.add(item=f"build:{bid}", status=status, role=pl["builds"][bid]["role"], s=round(time.time() - t0, 1),
                    error=None if status == "ok" else (se or so)[-1500:], notes=meta.get("notes"), loop=meta.get("loop"))
            done["ok" if status == "ok" else "failed"] += 1
            del procs[bid]
        time.sleep(0.05)
    print(f"[compile] shard {shard}/{nshards}: {done}", flush=True)
    return done


# a run's built objects: per build run/objs/<bid>.hsaco (+ .json), or the run's object index (one JSONL per writer,
# [bid, meta], meta["hsaco"] = the object in the shared cache): one append per build batch, no per-build files
_OBJIDX: Dict[str, tuple] = {}


def _obj_index(run: Path) -> Dict[str, dict]:
    d = run / "objs" / "index"
    if not d.is_dir():
        return {}
    files = sorted(d.glob("*.jsonl"))
    sig = tuple((p.name, p.stat().st_size) for p in files)
    hit = _OBJIDX.get(str(run))
    if hit and hit[0] == sig:
        return hit[1]
    idx: Dict[str, dict] = {}
    for p in files:
        for line in p.read_text().splitlines():
            try:
                b, m = json.loads(line)
                idx[b] = m
            except (ValueError, TypeError):
                pass  # a line cut by a killed writer
    _OBJIDX[str(run)] = (sig, idx)
    return idx


def obj_meta(run: Path, bid: str) -> Optional[dict]:
    """The build meta of ``bid`` in ``run`` (its .json file, else the run's object index)."""
    p = run / "objs" / f"{bid}.hsaco.json"
    if p.exists():
        return json.loads(p.read_text())
    return _obj_index(run).get(bid)


def obj_path(run: Path, bid: str) -> Optional[Path]:
    """The code object of ``bid`` in ``run`` (index entry, else run/objs/<bid>.hsaco), None when not built."""
    m = _obj_index(run).get(bid)
    if m and m.get("hsaco") and os.path.exists(m["hsaco"]):
        return Path(m["hsaco"])
    p = run / "objs" / f"{bid}.hsaco"
    return p if p.exists() else None


def build_rust(run: Path, pl: Mapping, bids: Sequence[str], cs: "CompileServer", led: "Ledger", done: dict) -> None:
    """The direct-compiler builds ``bids`` of a run through ``cs`` in ONE batch (every lever set): objects are linked
    from the shared cache into ``run/objs`` (symlink; a copy where links fail) with their meta JSON, and the ledger
    gets one append for the batch."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    groups: Dict[str, Dict[str, object]] = {}
    for bid in bids:
        groups.setdefault(pl["builds"][bid]["levers"], {})[bid] = pl["builds"][bid]["spec"]
    metas = cs.compile_multi(groups)
    per = (time.time() - t0) / max(1, len(bids))

    files = os.environ.get("LIBBENCH_OBJ_FILES", "0") == "1"
    index: Dict[str, dict] = {}

    def one(bid: str) -> dict:
        meta = metas.get(bid, {"ok": False, "error": "no result"})
        status = "ok" if meta.get("ok") else "failed"
        if status == "ok":
            index[bid] = meta
            if files:  # the per-build files too (readers outside obj_meta / obj_path)
                out = run / "objs" / f"{bid}.hsaco"
                _link(Path(meta["hsaco"]), out)
                Path(str(out) + ".json").write_text(json.dumps({**meta, "hsaco": str(out)}, indent=1, default=str))
            if _spills(meta):
                status = "spills"
        return dict(item=f"build:{bid}", status=status, role=pl["builds"][bid]["role"], s=round(per, 4),
                    error=meta.get("error"), notes=meta.get("notes"), loop=meta.get("loop"), server=True)

    t1 = time.time()
    if files:
        with ThreadPoolExecutor(128) as ex:  # file I/O on the shared home: overlap the round trips
            rows = list(ex.map(one, bids))
    else:
        rows = [one(b) for b in bids]
    if index:
        d = run / "objs" / "index"
        d.mkdir(parents=True, exist_ok=True)
        me = f"{os.uname().nodename if hasattr(os, 'uname') else 'local'}-{os.getpid()}"
        with open(d / f"{me}.jsonl", "a") as f:
            f.write("".join(json.dumps([b, m], default=str) + "\n" for b, m in index.items()))
    led.add_many(rows)
    for r in rows:
        done["ok" if r["status"] == "ok" else "failed"] += 1
    cs.last_stats["write_s"] = round(time.time() - t1, 2)


def _link(src: Path, dst: Path) -> None:
    """``dst`` -> ``src`` (atomic replace of any previous object)."""
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    try:
        os.symlink(os.path.realpath(src), tmp)
    except OSError:
        tmp.write_bytes(src.read_bytes())
    os.replace(tmp, dst)


# -------- build workers (a forked process pool: IR construction and object metadata are Python, GIL-bound)


def _ir_job(arg) -> Tuple[str, Optional[dict], Optional[str]]:
    """(spec key, spec dict, IR dir) -> (spec key, {ir_sha, ir, name, descriptor} | None, error | None)."""
    sk, sp, ir_dir = arg
    from rocke.core.ir_serialize import serialize

    g = _kmod()
    try:
        spec = replace(g.Mxfp6GemmSpec(), **dict(sp))
        ok, why = g.validate_spec(spec)
        if not ok:
            raise ValueError(why)
        g.check_compiler(spec, "rust")
        with _no_gc():
            text = serialize(g.build_mxfp6_gemm(spec, arch="gfx950"))
        name, desc = g.kernel_name(spec), g.descriptor(spec)
    except Exception as e:  # noqa: BLE001 -- a spec the builder refuses
        return sk, None, f"build: {e!r}"[-1500:]
    sha = hashlib.sha256(text.encode()).hexdigest()
    p = Path(ir_dir) / f"{sha}.ir"
    if not p.exists():
        tmp = Path(ir_dir) / f"{sha}.ir.tmp{os.getpid()}"
        tmp.write_text(text, newline="\n")
        os.replace(tmp, p)
    return sk, {"ir_sha": sha, "ir": str(p), "name": name, "descriptor": desc}, None


def _objmeta_job(arg) -> Tuple[str, dict]:
    """(object key, hsaco path) -> (key, {loop, notes}): the object-derived part of a build's meta."""
    key, hs_path = arg
    from benchmarks.gfx950.mxfp6_gemm import bench as B

    hs = Path(hs_path).read_bytes()
    return key, {"loop": B.isa_loop_stats(hs, Path(hs_path)), "notes": B.code_object_notes(hs)}


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, default=str))
    os.replace(tmp, path)


def _spills(meta: Mapping) -> bool:
    lp, nt = meta.get("loop") or {}, meta.get("notes") or {}
    return bool(nt.get("vgpr_spill_count") or nt.get("private_segment_fixed_size") or lp.get("scratch"))


# ================================================================== long-lived compile server (direct compiler)


class CompileServer:
    """Long-lived direct-compiler (Rust) build service. IR is built in THIS process: no process per build, since the
    LLVM history hazard does not apply to the direct compiler. Objects are cached by (IR sha, levers, compiler sha)
    and compiled with one ``target_compile`` invocation per lever set and batch of IR files (``jobs`` invocations
    in parallel). Reusable API (searches, WaveScope)::

        cs = CompileServer(tool, cache_dir)                            # cache may be shared across runs / jobs
        metas = cs.compile({"id1": spec_or_override_dict, ...}, levers="a,b")  # -> {id: meta}

    Meta matches ``bench.py compile``: ok, name, hsaco, notes, loop, rust_levers, rust_tool(_sha256), ir_sha256,
    error. LLVM builds are not served here (they keep the fresh-process path)."""

    def __init__(self, tool: str, cache: Path, jobs: int = 8, batch: int = 32, pool: Optional[bool] = None):
        self.tool, self.cache, self.jobs, self.batch = tool, Path(cache), jobs, batch
        # the forked IR / meta process pool: opt-in (compile_shard, the build service, LIBBENCH_BUILD_POOL=1); a
        # caller that already initialized the GPU runtime keeps the in-process path (no fork after HIP init)
        self.use_pool = bool(pool if pool is not None else os.environ.get("LIBBENCH_BUILD_POOL") == "1") and jobs > 1
        self.cache.mkdir(parents=True, exist_ok=True)
        self.tool_sha = hashlib.sha256(Path(tool).read_bytes()).hexdigest()
        self.known = compiler_levers(tool)
        # spec -> {ir_sha, name, descriptor}, valid for one staged tree (LIBBENCH_TREE_SHA, set by the job env from
        # the stage tarball's sha256); without a tree id every spec rebuilds its IR (objects stay cached by IR sha)
        self.tree = os.environ.get("LIBBENCH_TREE_SHA", "")
        self.ircache = Path(os.environ.get("LIBBENCH_IR_CACHE", str(self.cache.parent / "ircache")))
        self.ircache.mkdir(parents=True, exist_ok=True)
        self.pool = None  # the forked process pool of compile_multi (kept for the session)
        self.last_stats: Dict[str, object] = {}
        if self.use_pool:
            # fork the workers now, before the compiler processes and their pipe threads exist
            list(self._pool().map(abs, range(self.jobs)))
        self.rc = None  # ``jobs`` long-lived ``target_compile --serve`` processes (compiler builds >= out-68371726)
        client = Path(tool).resolve().parent.parent / "tools" / "rc_server.py"
        if client.exists() and os.environ.get("LIBBENCH_RC_SERVE", "1") == "1":
            import importlib.util

            spec = importlib.util.spec_from_file_location("rc_server", client)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.rc = mod.CompileServer(tool, workers=max(1, jobs))

    def close(self) -> None:
        if self.rc is not None:
            self.rc.close()
            self.rc = None
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    def _pool(self):
        if self.pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            from benchmarks.gfx950.mxfp6_gemm import bench as _B  # noqa: F401 -- imported before the fork
            from rocke.core.ir_serialize import serialize as _s  # noqa: F401

            _kmod()
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
            self.pool = ProcessPoolExecutor(max(1, self.jobs), mp_context=ctx)
        return self.pool

    # -------- append-only cache indexes (one JSONL per writer process; every writer's lines merged on load)

    def _idx_load(self, d: Path) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if d.is_dir():
            for p in sorted(d.glob("*.jsonl")):
                for line in p.read_text().splitlines():
                    try:
                        k, v = json.loads(line)
                        out[k] = v
                    except (ValueError, TypeError):
                        pass  # a line cut by a killed writer
        return out

    def _idx_append(self, d: Path, rows: Mapping[str, dict]) -> None:
        if rows:
            d.mkdir(parents=True, exist_ok=True)
            me = f"{os.uname().nodename if hasattr(os, 'uname') else 'local'}-{os.getpid()}"
            with open(d / f"{me}.jsonl", "a") as f:
                f.write("".join(json.dumps([k, v], default=str) + "\n" for k, v in rows.items()))

    def _indexes(self):
        """(spec -> IR record index of this tree, object meta index), loaded once per process and refreshed when
        another writer added lines (the build service lives across many requests)."""
        ird = self.ircache / (self.tree[:16] if self.tree else "notree")
        md = self.cache / "meta"
        sig = tuple((p.name, p.stat().st_size) for d in (ird, md) if d.is_dir() for p in sorted(d.glob("*.jsonl")))
        if getattr(self, "_idx_sig", None) != sig:
            self._ir_idx = self._idx_load(ird) if self.tree else {}
            self._meta_idx = self._idx_load(md)
            self._idx_sig = sig
        return ird, md

    def compile_multi(self, groups: Mapping[str, Mapping[str, object]]) -> Dict[str, dict]:
        """``{levers: {id: spec}}`` -> ``{id: meta}``, every lever set in one streaming pass:
          IR    -- specs not in the IR index are built in the forked process pool (``jobs`` workers);
          object-- as each IR completes, its (IR, levers, compiler) objects not in the cache go to the next free
                   long-lived compiler process (one request per object, own output dir);
          meta  -- as each object lands, its ISA loop stats and code-object notes are computed in the pool.
        The three phases overlap. Caches: objects by key in ``self.cache``; object meta and the spec -> IR record
        (per staged tree) in append-only indexes. Phase times and hit counts in ``self.last_stats``."""
        import shutil
        import tempfile
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        st = {"items": 0, "ir_hit": 0, "ir_built": 0, "obj_hit": 0, "obj_built": 0, "meta_hit": 0, "meta_built": 0}
        t0 = time.time()
        ird, md = self._indexes()
        out: Dict[str, dict] = {}
        items = []  # (id, levers, spec dict, spec key)
        for levers, its in groups.items():
            unk = unknown_levers(self.known, levers)
            if unk is None or unk:
                why = "cannot list its levers" if unk is None else f"unknown levers {unk}"
                for i in its:
                    out[i] = {"ok": False, "error": f"direct compiler {self.tool}: {why}"}
                continue
            for iid, sp in its.items():
                sd = asdict(sp) if hasattr(sp, "__dataclass_fields__") else dict(sp)
                items.append((iid, levers, sd, _sha({"spec": sd, "tree": self.tree or f"pid{os.getpid()}"}, 24)))
        st["items"] = len(items)

        def objkey(ir_sha: str, levers: str) -> str:
            return _sha({"ir": ir_sha, "levers": levers, "tool": self.tool_sha}, 24)

        levs_of: Dict[str, set] = {}
        for _, levers, _, sk in items:
            levs_of.setdefault(sk, set()).add(levers)
        irs: Dict[str, dict] = {}
        ir_err: Dict[str, str] = {}
        obj_err: Dict[str, str] = {}
        om: Dict[str, dict] = {}
        new_ir: Dict[str, dict] = {}
        new_meta: Dict[str, dict] = {}
        lock = threading.Lock()
        scheduled: set = set()
        meta_futs = []
        shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
        stamps = {"ir_done": t0, "obj_done": t0}
        with tempfile.TemporaryDirectory(dir=shm) as td:
            tp = ThreadPoolExecutor(max(1, len(self.rc.workers) if self.rc is not None else self.jobs))
            obj_futs = []

            def meta_of(k: str) -> None:
                if k in om:
                    return
                if k in self._meta_idx:
                    om[k] = self._meta_idx[k]
                    st["meta_hit"] += 1
                    return
                meta_futs.append(self._submit(_objmeta_job, (k, str(self.cache / f"{k}.hsaco"))))

            def build_obj(k: str, ir: str, levers: str) -> str:
                try:
                    return _build_obj(k, ir, levers)
                except Exception as e:  # noqa: BLE001 -- one failed object, not the batch
                    obj_err[k] = f"direct compiler: {e!r}"[-1500:]
                    return k

            def _build_obj(k: str, ir: str, levers: str) -> str:
                od = Path(td) / "o" / k
                od.mkdir(parents=True, exist_ok=True)
                if self.rc is not None:
                    r = self.rc.compile("gfx950", str(od), ir, levers)
                    ok, h, err = r.get("ok"), r.get("hsaco"), r.get("error", "?")
                else:
                    pr = subprocess.run([self.tool, "gfx950", str(od), ir], capture_output=True, text=True,
                                        timeout=1800, env={**os.environ, "ROCKE_DIRECT_LEVERS": levers})
                    h = str(od / (Path(ir).stem + ".hsaco"))
                    ok, err = os.path.exists(h), (pr.stderr or pr.stdout)
                if ok and h and os.path.exists(h):
                    tmp = self.cache / f"{k}.hsaco.tmp{os.getpid()}.{threading.get_ident()}"
                    shutil.copyfile(h, tmp)
                    os.replace(tmp, self.cache / f"{k}.hsaco")
                    with lock:
                        meta_of(k)
                else:
                    obj_err[k] = f"direct compiler: {err}"[-1500:]
                return k

            def on_ir(sk: str) -> None:  # schedule this spec's objects (called with the lock held)
                for levers in levs_of[sk]:
                    k = objkey(irs[sk]["ir_sha"], levers)
                    if k in scheduled:
                        continue
                    if (self.cache / f"{k}.hsaco").exists():
                        scheduled.add(k)
                        st["obj_hit"] += 1
                        meta_of(k)
                    elif irs[sk].get("ir"):
                        scheduled.add(k)
                        st["obj_built"] += 1
                        obj_futs.append(tp.submit(build_obj, k, irs[sk]["ir"], levers))
                    elif sk not in need:
                        need.append(sk)  # an indexed IR whose object is missing: rebuild the IR text

            need: List[str] = []
            for _, levers, sd, sk in items:
                if sk in irs or sk in need:
                    continue
                rec = self._ir_idx.get(sk) if self.tree else None
                if rec:
                    irs[sk] = dict(rec)
                    st["ir_hit"] += 1
                else:
                    need.append(sk)
            with lock:
                for sk in list(irs):
                    on_ir(sk)
            spec_of = {sk: sd for _, _, sd, sk in items}
            ir_futs = [self._submit(_ir_job, (sk, spec_of[sk], td)) for sk in dict.fromkeys(need)]
            for fut in as_completed(ir_futs):
                sk, rec, err = fut.result()
                with lock:
                    if err:
                        ir_err[sk] = err
                        continue
                    hit = sk in irs
                    irs[sk] = rec
                    if not hit:
                        st["ir_built"] += 1
                        new_ir[sk] = {k: v for k, v in rec.items() if k != "ir"}
                    on_ir(sk)
            stamps["ir_done"] = time.time()
            for f in as_completed(obj_futs):
                f.result()
            stamps["obj_done"] = time.time()
            tp.shutdown(wait=True)
            for f in meta_futs:
                k, m = f.result()
                om[k] = m
                new_meta[k] = m
                st["meta_built"] += 1
        if self.tree:
            self._idx_append(ird, new_ir)
            self._ir_idx.update(new_ir)
        self._idx_append(md, new_meta)
        self._meta_idx.update(new_meta)
        for iid, levers, sd, sk in items:
            if iid in out:
                continue
            if sk in ir_err:
                out[iid] = {"ok": False, "error": ir_err[sk]}
                continue
            r = irs[sk]
            k = objkey(r["ir_sha"], levers)
            if k in obj_err or k not in om:
                out[iid] = {"ok": False, "error": obj_err.get(k, "no object")}
                continue
            out[iid] = {"ok": True, "name": r["name"], "hsaco": str(self.cache / f"{k}.hsaco"), "compiler": "rust",
                        "loop": om[k]["loop"], "notes": om[k]["notes"], "rust_levers": levers, "rust_tool": self.tool,
                        "rust_tool_sha256": self.tool_sha, "ir_sha256": r["ir_sha"], "cache_key": k,
                        "descriptor": r["descriptor"]}
        t_end = time.time()
        st.update(ir_s=round(stamps["ir_done"] - t0, 2), obj_s=round(stamps["obj_done"] - t0, 2),
                  total_s=round(t_end - t0, 2))
        self.last_stats = st
        return out

    def _submit(self, fn, arg):
        """``fn(arg)`` in the process pool (inline when the pool is off: a completed future)."""
        if not self.use_pool:
            from concurrent.futures import Future

            f = Future()
            try:
                f.set_result(fn(arg))
            except Exception as e:  # noqa: BLE001
                f.set_exception(e)
            return f
        return self._pool().submit(fn, arg)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _meta(self, hs_path: Path, spec, key: str, levers: str, ir_sha: str) -> dict:
        from benchmarks.gfx950.mxfp6_gemm import bench as B

        g = _kmod()
        hs = hs_path.read_bytes()
        return {"ok": True, "name": g.kernel_name(spec), "hsaco": str(hs_path), "compiler": "rust",
                "loop": B.isa_loop_stats(hs, hs_path), "notes": B.code_object_notes(hs), "rust_levers": levers,
                "rust_tool": self.tool, "rust_tool_sha256": self.tool_sha, "ir_sha256": ir_sha, "cache_key": key,
                "descriptor": g.descriptor(spec)}

    def compile(self, items: Mapping[str, object], levers: str) -> Dict[str, dict]:
        """``{id: spec}`` with one lever set -> ``{id: meta}`` (``compile_multi``)."""
        return self.compile_multi({levers: items})

    def _map(self, fn, args):
        if not self.use_pool or len(args) <= 1:
            return list(map(fn, args))
        return list(self._pool().map(fn, args, chunksize=max(1, len(args) // (4 * self.jobs))))

    def _compile_serial(self, items: Mapping[str, object], levers: str) -> Dict[str, dict]:
        """The previous single-process path (kept for comparison: LIBBENCH_SERIAL_BUILD=1 in compile_shard)."""
        import tempfile
        from concurrent.futures import ThreadPoolExecutor

        from rocke.core.ir_serialize import serialize

        g = _kmod()
        unk = unknown_levers(self.known, levers)
        if unk is None or unk:
            why = "cannot list its levers" if unk is None else f"unknown levers {unk}"
            return {i: {"ok": False, "error": f"direct compiler {self.tool}: {why}"} for i in items}
        out: Dict[str, dict] = {}
        todo: Dict[str, list] = {}  # key -> [ir text, spec, ir sha, [ids]]
        for iid, sp in items.items():
            spec = sp if isinstance(sp, g.Mxfp6GemmSpec) else replace(g.Mxfp6GemmSpec(), **dict(sp))
            try:
                ok, why = g.validate_spec(spec)
                if not ok:
                    raise ValueError(why)
                g.check_compiler(spec, "rust")
                with _no_gc():
                    text = serialize(g.build_mxfp6_gemm(spec, arch="gfx950"))
            except Exception as e:  # noqa: BLE001 -- a spec the builder refuses
                out[iid] = {"ok": False, "error": f"build: {e!r}"[-1500:]}
                continue
            ir_sha = hashlib.sha256(text.encode()).hexdigest()
            key = _sha({"ir": ir_sha, "levers": levers, "tool": self.tool_sha}, 24)
            hs = self.cache / f"{key}.hsaco"
            if hs.exists():
                out[iid] = self._meta(hs, spec, key, levers, ir_sha)
                continue
            todo.setdefault(key, [text, spec, ir_sha, []])[3].append(iid)
        keys = list(todo)

        def run_batch(ks: List[str]) -> None:
            with tempfile.TemporaryDirectory() as td:
                files = []
                for k in ks:
                    p = Path(td) / f"{k}.ir"
                    p.write_text(todo[k][0], newline="\n")
                    files.append(str(p))
                r = subprocess.run([self.tool, "gfx950", td] + files, capture_output=True, text=True, timeout=1800,
                                   env={**os.environ, "ROCKE_DIRECT_LEVERS": levers})
                for k in ks:
                    h = Path(td) / f"{k}.hsaco"
                    if h.exists():
                        tmp = self.cache / f"{k}.hsaco.tmp{os.getpid()}"
                        tmp.write_bytes(h.read_bytes())
                        os.replace(tmp, self.cache / f"{k}.hsaco")  # atomic: concurrent servers share the cache
                    else:
                        for iid in todo[k][3]:
                            out[iid] = {"ok": False, "error": f"direct compiler: {(r.stderr or r.stdout)[-1500:]}"}

        if self.rc is not None and keys:  # long-lived compiler processes, one request per IR (pipelined)
            shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
            with tempfile.TemporaryDirectory(dir=shm) as td:
                reqs = []
                for k in keys:
                    ip = Path(td) / f"{k}.ir"
                    ip.write_text(todo[k][0], newline="\n")
                    reqs.append(("gfx950", td, str(ip), levers))
                for k, r in zip(keys, self.rc.compile_many(reqs)):
                    if r.get("ok") and os.path.exists(r["hsaco"]):
                        tmp = self.cache / f"{k}.hsaco.tmp{os.getpid()}"
                        tmp.write_bytes(Path(r["hsaco"]).read_bytes())
                        os.replace(tmp, self.cache / f"{k}.hsaco")
                    else:
                        for iid in todo[k][3]:
                            out[iid] = {"ok": False, "error": f"direct compiler: {r.get('error', '?')}"[-1500:]}
        else:
            batches = [keys[i:i + self.batch] for i in range(0, len(keys), self.batch)]
            with ThreadPoolExecutor(max(1, self.jobs)) as ex:
                list(ex.map(run_batch, batches))
        for k in keys:
            text, spec, ir_sha, ids = todo[k]
            hs = self.cache / f"{k}.hsaco"
            if hs.exists():
                m = self._meta(hs, spec, k, levers, ir_sha)
                for iid in ids:
                    out[iid] = m
        return out


# ================================================================== build service (long-lived, one CPU node)


def _read_current(path: Optional[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def buildd(queue: Path, jobs: int, current: Optional[Path] = None, idle_exit_s: float = 0.0,
           poll_s: float = 0.25, http: bool = True, root: Optional[Path] = None, bind: str = "0.0.0.0") -> int:
    """Build service: ONE warm compile session (forked IR / meta pool + ``jobs`` long-lived compiler processes, the
    lane caches) serving every lane's build requests, so nobody pays job start, imports or untar per batch.

    Requests (JSON):
      {"type": "run", "run": <run dir>, "manifest": <manifest path, optional: plan first>, "stage_sha": ..}
      {"type": "specs", "levers": "..", "items": {id: spec}, "out": <dir: one <id>.json meta each>, "stage_sha": ..}
    Two channels, served one request at a time (one compile session):
      HTTP (default): POST the request to the ``url`` in ``<queue>/HEARTBEAT`` with header ``X-Token: <token>``
        (HEARTBEAT is mode 600); the response is the result JSON (no shared-filesystem polling: NFS attribute
        caching delays a file answer by tens of seconds across nodes);
      files: ``<queue>/<name>.req`` (write ``.tmp`` then rename); answered ``<name>.done`` / ``.bad`` / ``.stale``.
    Paths in a request must lie under ``root`` (default: the queue's parent, the lane dir). Staleness: the service
    runs the lane's current tree (``current``: the CURRENT pointer); when it moves the service exits 75 (its wrapper
    restarts it on the new tree; pending requests wait); a request whose stage_sha is not the current one is
    answered stale. ``<queue>/stop`` ends the service. Returns the exit code."""
    import queue as _q
    import secrets
    import threading

    queue.mkdir(parents=True, exist_ok=True)
    root = Path(root or queue.parent).resolve()
    tree = os.environ.get("LIBBENCH_TREE_SHA", "")
    tool = os.environ.get("ROCKE_RUST_COMPILE", "")
    cache = Path(os.environ.get("LIBBENCH_OBJ_CACHE", str(queue / "objcache")))
    me = {"host": os.uname().nodename if hasattr(os, "uname") else "local", "job": os.environ.get("SLURM_JOB_ID"),
          "pid": os.getpid(), "tree": tree, "compiler": tool, "jobs": jobs}
    t_start = time.time()
    cs = CompileServer(tool, cache, jobs=jobs, pool=True) if tool and os.path.exists(tool) and not _fake() else None
    me["ready_s"] = round(time.time() - t_start, 2)
    state = {"cur": _read_current(current), "busy": False}

    def under_root(p) -> Path:
        q = Path(p).resolve()
        if q != root and root not in q.parents:
            raise ValueError(f"path {p} is outside the service root {root}")
        return q

    def process(req: Mapping) -> dict:
        t0 = time.time()
        cur = state["cur"]
        if req.get("stage_sha") and cur.get("stage_sha") and req["stage_sha"] != cur["stage_sha"]:
            return {"stale": True, "error": f"stale: superseded by {cur['stage_sha'][:16]}",
                    "request_stage": req["stage_sha"][:16]}
        if req.get("type") == "run":
            run = under_root(req["run"])
            if req.get("manifest"):
                plan(load_manifest(str(under_root(req["manifest"]))), run)
            r = compile_shard(run, 0, 1, jobs, bool(req.get("redo_failed")), cs=cs)
            res = {"ok": True, **r, "phases": cs.last_stats if cs else {}}
        elif req.get("type") == "specs":
            if cs is None:
                raise RuntimeError("no direct compiler in this service")
            od = under_root(req["out"])
            metas = cs.compile_multi({req.get("levers", ""): req["items"]})
            od.mkdir(parents=True, exist_ok=True)
            for iid, m in metas.items():
                _atomic_json(od / f"{iid}.json", m)
            res = {"ok": True, "items": len(metas), "built_ok": sum(1 for m in metas.values() if m.get("ok")),
                   "phases": cs.last_stats}
        else:
            raise ValueError(f"unknown request type {req.get('type')!r}")
        res.update(s=round(time.time() - t0, 2), service=me)
        return res

    memq: "_q.Queue" = _q.Queue()
    token = secrets.token_hex(16)
    srv = None
    if http:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class _H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                if self.headers.get("X-Token") != token:
                    self._send(403, {"error": "bad token"})
                    return
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                except ValueError as e:
                    self._send(400, {"error": f"bad JSON: {e}"})
                    return
                box: Dict[str, object] = {}
                ev = threading.Event()
                memq.put((req, box, ev))
                ev.wait()
                self._send(200, box["res"])

            def do_GET(self):  # noqa: N802 -- liveness, no token needed (no work)
                self._send(200, {**me, "busy": state["busy"], "t": time.time()})

            def _send(self, code: int, obj) -> None:
                data = json.dumps(obj, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):  # quiet
                pass

        srv = ThreadingHTTPServer((bind, 0), _H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        me["url"] = f"http://{socket.getfqdn() if bind == '0.0.0.0' else bind}:{srv.server_address[1]}"
    print(f"[buildd] ready in {me['ready_s']} s: {me}", flush=True)

    hb_last = {"t": 0.0, "busy": None}

    def heartbeat(busy) -> None:
        # rewritten at most every 5 s (or when busy changes): readers on other NFS clients get ESTALE while a file
        # they opened is replaced, so frequent replaces starve them
        if busy == hb_last["busy"] and time.time() - hb_last["t"] < 5.0:
            return
        hb_last.update(t=time.time(), busy=busy)
        hb = queue / "HEARTBEAT"
        tmp = hb.with_name(f"HEARTBEAT.tmp{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({**me, "token": token, "t": time.time(), "busy": busy}, f)
        os.replace(tmp, hb)

    last_work = time.time()
    sd = os.environ.get("LIBBENCH_STAGE_DIR", "")
    last_touch = last_scan = 0.0
    try:
        while True:
            heartbeat(False)
            if sd and time.time() - last_touch > 600:  # keep this node's unpacked tree marked in use (pruning)
                Path(sd, ".ok").touch()
                last_touch = time.time()
            if (queue / "stop").exists():
                return 0
            state["cur"] = _read_current(current)
            if tree and state["cur"].get("stage_sha") and state["cur"]["stage_sha"] != tree:
                print(f"[buildd] tree superseded by {state['cur']['stage_sha'][:16]}: restarting", flush=True)
                return 75
            item = None
            try:
                item = memq.get(timeout=poll_s)
            except _q.Empty:
                pass
            if item is None and time.time() - last_scan >= poll_s:
                last_scan = time.time()
                reqs = sorted(queue.glob("*.req"), key=lambda p: p.stat().st_mtime)
                if reqs:
                    rq = reqs[0]
                    name = rq.name[:-4]
                    claimed = queue / f"{name}.run"
                    try:
                        os.replace(rq, claimed)
                        item = ("file", name, claimed)
                    except OSError:
                        pass  # another service claimed it
            if item is None:
                if idle_exit_s and time.time() - last_work > idle_exit_s:
                    return 0
                continue
            state["busy"] = True
            heartbeat(item[1] if item[0] == "file" else "http")
            try:
                if item[0] == "file":
                    _, name, claimed = item
                    try:
                        res = process(json.loads(claimed.read_text()))
                        kind = "stale" if res.get("stale") else "done"
                    except Exception as e:  # noqa: BLE001 -- a bad request must not stop the service
                        res, kind = {"error": repr(e)[-2000:]}, "bad"
                    _atomic_json(queue / f"{name}.{kind}", res)
                    try:
                        claimed.rename(queue / f"{name}.req.served")
                    except OSError:
                        pass
                else:
                    req, box, ev = item
                    try:
                        box["res"] = process(req)
                    except Exception as e:  # noqa: BLE001
                        box["res"] = {"ok": False, "error": repr(e)[-2000:]}
                    ev.set()
            finally:
                state["busy"] = False
                last_work = time.time()
    finally:
        if srv is not None:
            srv.shutdown()
        if cs is not None:
            cs.close()


# ================================================================== GPU stage


def occupancy_of(notes: Mapping, waves: int) -> int:
    """Work-groups per CU from the code object's resource notes (VGPR + AGPR per lane, LDS per WG)."""
    regs = int(notes.get("vgpr_count") or 0) + int(notes.get("agpr_count") or 0)
    lds = int(notes.get("group_segment_fixed_size") or 0)
    if regs <= 0:
        return 0
    per_simd = T.REGS_PER_LANE // (8 * -(-regs // 8))
    occ_regs = per_simd * T.SIMD_PER_CU // max(waves, 1)
    occ_lds = T.LDS_BYTES // lds if lds else 16
    return max(0, min(occ_regs, occ_lds))


class Data:
    """Host operands per (dist, fmt, rows, k, source) with a disk cache, and float64 references per entry shape."""

    def __init__(self, cache: Path, seed: int, workers: int):
        self.cache, self.seed, self.workers = cache, seed, workers
        self.mem: Dict[tuple, object] = {}

    def operands(self, dist: str, fmt: str, m: int, n: int, k: int, src: int):
        from benchmarks.gfx950.mxfp6_gemm import bench as B

        key = ("op", dist, fmt, m, n, k, src)
        if key not in self.mem:
            if dist.startswith("order"):  # split-K order set ("order<s>": skbench.order_operands; every source alike)
                from builders.gfx950.mxfp6_gemm.skbench import order_operands

                a, b, want = order_operands(m, n, k, int(dist[5:]), fmt)
                self.mem[("want", dist, fmt, m, n, k)] = want
                self.mem[key] = (a, b)
            else:
                self.mem[key] = B.cached_operands(self.cache, m, n, k, dist, self.seed + 1000 * src, self.workers,
                                                  fmt)
        return self.mem[key]

    def reference(self, dist: str, fmt: str, m: int, n: int, k: int, src: int):
        from benchmarks.gfx950.mxfp6_gemm import bench as B

        key = ("ref", dist, fmt, m, n, k, src)
        if key not in self.mem and dist.startswith("order"):
            # the order set's expected output: the float32 fold of the exact per-split partials in split order
            # (checked bit-exact, _check_cell); not the float64 product
            self.operands(dist, fmt, m, n, k, src)
            want = self.mem[("want", dist, fmt, m, n, k)].astype(np.float64)
            self.mem[key] = (np.broadcast_to(want[None, :], (m, n)).copy(), np.zeros((m, n)))
        if key not in self.mem:
            a, b = self.operands(dist, fmt, m, n, k, src)
            sfx = "" if fmt == "e2m3" else f"_{fmt}"
            self.mem[key] = B.cached_reference(self.cache, a, b, f"{dist}_{m}x{n}x{k}_s{self.seed + 1000 * src}{sfx}")
        return self.mem[key]


def kernargs(mod, spec, ptrs: Mapping[str, int], prob: Mapping[str, int], strides: Mapping[str, int]) -> bytes:
    """Kernel arguments of a build: the kernel module's own ``kernargs`` helper when it has one, else the static
    form ``a, b, sa, sb, c`` (anything else is refused: the harness never guesses an argument layout)."""
    if hasattr(mod, "kernargs"):
        return mod.kernargs(spec, **ptrs, **{k: v for k, v in prob.items() if k in ("m", "n", "k", "batch")},
                            **strides)
    params = list(mod.descriptor(spec).get("params", []))
    if params != ["a", "b", "sa", "sb", "c"]:
        raise RuntimeError(f"kernel params {params}: the kernel module needs a kernargs(spec, a, b, sa, sb, c, m, n, k, "
                           "batch, **strides) helper for this harness")
    return struct.pack("<QQQQQ", ptrs["a"], ptrs["b"], ptrs["sa"], ptrs["sb"], ptrs["c"])


class GpuCell:
    """Device state of one cell: per (dist, set index) input buffers, the C buffer, loaded arms."""

    def __init__(self, rt, L, mod, data: Data, cell: Mapping, man: Mapping):
        self.rt, self.L, self.mod, self.data, self.cell, self.man = rt, L, mod, data, cell, man
        p = cell["problem"]
        self.batch, self.m, self.n, self.k = int(p["batch"]), int(p["m"]), int(p["n"]), int(p["k"])
        self.flop = 2.0 * self.batch * self.m * self.n * self.k
        self.c = rt.alloc(self.batch * self.m * self.n * 2)
        self.c_host = np.empty((self.batch, self.m, self.n), dtype=np.uint16)
        self.bufs: Dict[tuple, dict] = {}
        self.allocs: List[int] = [self.c]

    def _put(self, arr: np.ndarray) -> int:
        import ctypes

        arr = np.ascontiguousarray(arr)
        ptr = self.rt.alloc(arr.nbytes)
        self.rt.memcpy_h2d(ptr, (ctypes.c_ubyte * int(arr.nbytes)).from_buffer(arr), arr.nbytes)
        self.allocs.append(ptr)
        return ptr

    def inputs(self, spec, dist: str, n_src: int, set_idx: int = 0) -> dict:
        """Pointers + per-entry strides: entry e holds source e mod n_src, packed in ``spec``'s layouts and
        concatenated (packed per-entry strides)."""
        key = (dist, n_src, set_idx, self.mod.operand_layout_of(spec, "A"), self.mod.operand_layout_of(spec, "B"),
               self.mod.scale_layout_of(spec, "A"), self.mod.scale_layout_of(spec, "B"), spec.tile_m, spec.tile_n,
               spec.atom) + ((True,) if getattr(spec, "m_mask", False) else ()) + (
                   ("kt",) if getattr(spec, "k_tail", False) else ())  # m_mask: A scales padded; k_tail: K padded
        if key in self.bufs:
            return self.bufs[key]
        L = self.L
        la, lb, sla, slb = key[3:7]
        imgs = []
        for s in range(n_src):
            a, b = self.data.operands(dist, spec.elem_fmt, self.m, self.n, self.k, s)
            sa = a.scales
            if getattr(spec, "k_tail", False):  # K % 128 != 0 (SB-5): B and both scale images padded to whole 128-K
                b = L.pad_k(b)                   # tiles (zero codes / scales); A's tail is bounded by its descriptor
                sa = L.pad_k(a).scales
            if getattr(spec, "m_mask", False) and a.rows % spec.tile_m:  # splitk m_mask: whole tile rows, zero-padded
                sa = np.concatenate([sa, np.zeros((spec.tile_m - a.rows % spec.tile_m, sa.shape[1]), dtype=np.uint8)])
            imgs.append((L.pack_operand(a, la, spec.tile_m), L.pack_operand(b, lb, spec.tile_n),
                         L.pack_scales_for(sa, spec.atom, sla), L.pack_scales_for(b.scales, spec.atom, slb)))
        cat = [np.concatenate([imgs[e % n_src][i].view(np.uint8).reshape(-1) for e in range(self.batch)])
               for i in range(4)]
        strides = {"stride_a": imgs[0][0].nbytes, "stride_b": imgs[0][1].nbytes, "stride_sa": imgs[0][2].nbytes,
                   "stride_sb": imgs[0][3].nbytes, "stride_c": self.m * self.n * 2}
        ptrs = {"a": self._put(cat[0]), "b": self._put(cat[1]), "sa": self._put(cat[2]), "sb": self._put(cat[3]),
                "c": self.c}
        self.bufs[key] = {"ptrs": ptrs, "strides": strides}
        return self.bufs[key]

    def sk_buffers(self, arm: str, spec) -> dict:
        """splitk: the arm's own workspace and counters (a mutant's counters never reach another arm's launches)."""
        if not hasattr(self, "_sk"):
            self._sk = {}
        if arm not in self._sk:
            from kernels.gfx950.mxfp6_gemm import splitk as SK

            wsb, cb = SK.buffers(spec, self.m, self.n, self.k, self.batch)
            ws, cnt = self.rt.alloc(wsb), self.rt.alloc(cb)
            self.rt.memset(cnt, 0, cb)
            self.allocs += [ws, cnt]
            self._sk[arm] = {"ws": ws, "cnt": cnt}
        return self._sk[arm]

    def run_once(self, fn, grid, block, ka: bytes) -> np.ndarray:
        import ctypes

        self.rt.memset(self.c, 0xFF, self.c_host.nbytes)  # NaN fill
        self.rt.launch(fn, grid, block, ka)
        self.rt.wait_stream(0)
        self.rt.memcpy_d2h((ctypes.c_ubyte * int(self.c_host.nbytes)).from_buffer(self.c_host), self.c, self.c_host.nbytes)
        return self.c_host.copy()

    def free(self) -> None:
        for p in self.allocs:
            try:
                self.rt.free(p)
            except Exception:  # noqa: BLE001 -- best effort; the process ends after the shard anyway
                pass


def _check_cell(L, data: Data, out: np.ndarray, dist: str, fmt: str, m: int, n: int, k: int, n_src: int) -> dict:
    """Every entry against its source's float64 reference (integer set bit-exact)."""
    worst, bad, n_tot, finite = 0.0, 0, 0, True
    bad_entries = []
    for e in range(out.shape[0]):
        rc, rs = data.reference(dist, fmt, m, n, k, e % n_src)
        r = L.check_against_reference(out[e], rc, rs, exact=(dist == "int" or dist.startswith("order")))
        bad += r["bad"]
        n_tot += r["n"]
        finite &= r["finite"]
        worst = max(worst, r.get("worst_err_over_tol", 0.0))
        if r["bad"]:
            bad_entries.append(e)
    return {"pass": bad == 0 and finite, "bad": bad, "n": n_tot, "finite": finite, "worst_err_over_tol": worst,
            "bad_entries": bad_entries[:16], "exact": dist == "int" or dist.startswith("order")}


def _fake_cell(cell: Mapping, rng: random.Random) -> dict:
    t = T.Tile.parse(cell["tile"])
    # unit-free placeholder values (pipeline self-test only, not measurements)
    tf = min(1.0, (t.mt0 * t.mt1) / 65536.0) ** 0.25 * rng.uniform(0.98, 1.02)
    warm = [{"tflops": tf * rng.uniform(0.995, 1.005), "ms": 1.0, "sclk_med": 1.0, "watts_med": 1.0, "mj": 1.0}
            for _ in range(3)]
    return {"verify": {"ok": True, "fake": True}, "timing": {
        "arms": {"cand": {"warm": warm, "cold": [tf * 0.97] * 5}, "twin": {"warm": warm, "cold": [tf * 0.97] * 5},
                 "probe": {"warm": [{"tflops": 1.5, "sclk_med": 1.0, "watts_med": 1.0}], "cold": [1.4]}},
        "twin_ratio_warm": {"median": 1.0, "lo": 0.998, "hi": 1.002, "n": 3},
        "cand_warm_tflops": tf, "cand_cold_tflops": tf * 0.97, "cand_warm_mj": 1.0, "cand_warm_sclk": 1.0,
        "cand_warm_watts": 1.0, "probe_warm_tflops": 1.5}}


def _noisy(run: Path, cid: str) -> bool:
    """A done cell whose A-vs-A twin gate failed (re-time it with ``redo_noisy``)."""
    p = run / "cells" / f"{_sha(cid)}.json"
    if not p.exists():
        return False
    t = json.loads(p.read_text()).get("timing") or {}
    return t.get("twin_gate") is False


class GpuSession:
    """A long-lived GPU worker context for the visible GPU: one HIP runtime, the SCLK / power sampler, and the host
    operand / float64 reference cache (in memory + node-local disk), reused by every ``gpu_shard`` call of the
    process. Reusable API (searches, WaveScope)::

        ses = GpuSession(cache_dir, seed=42)
        gpu_shard(run, cells=[...], session=ses)      # as many calls / runs as needed in this process
    or ``python -m builders.gfx950.mxfp6_gemm.libbench serve --run R --inbox DIR`` (requests as files, below)."""

    def __init__(self, cache: Path, seed: int = 42, workers: int = 16):
        from benchmarks.gfx950.mxfp6_gemm import bench as B
        from builders.gfx950.mxfp6_gemm import tune as TU
        from kernels.gfx950.mxfp6_gemm import layout as L
        from rocke.runtime.hip_module import Runtime

        self.B, self.TU, self.L, self.g = B, TU, L, _kmod()
        self.rt = Runtime()
        self.bdf = B.device_bdf()
        self.clk = TU.Clock(self.bdf)
        self.node = os.environ.get("SLURMD_NODENAME", "")
        self._data: Dict[tuple, Data] = {}
        self.data_for(cache, seed, workers)

    def data_for(self, cache: Path, seed: int, workers: int) -> "Data":
        key = (str(cache), seed)
        if key not in self._data:
            self._data[key] = Data(Path(cache), seed, workers)
        return self._data[key]


def serve(run_root: Path, inbox: Path, cache: Path, workers: int = 16, poll_s: float = 0.5) -> int:
    """Persistent GPU worker: one process per GPU, alive across many requests. A request is a JSON file
    ``<inbox>/<name>.req`` = {"run": run dir, "cells": [ids] or "cells_file": path, "redo_failed": bool}; the worker
    runs it with the shared ``GpuSession`` and writes ``<name>.done`` ({"done": n} or {"error": ...}). ``<inbox>/stop``
    ends it. A HIP fault exits 86 (the caller restarts the worker; the faulting cell is in the ledger)."""
    inbox.mkdir(parents=True, exist_ok=True)
    ses = None if _fake() else GpuSession(cache, workers=workers)
    print(f"[serve] {getattr(ses, 'node', 'fake')} {getattr(ses, 'bdf', 'fake')} inbox {inbox}", flush=True)
    while not (inbox / "stop").exists():
        reqs = sorted(inbox.glob("*.req"))
        if not reqs:
            time.sleep(poll_s)
            continue
        rq = reqs[0]
        work = rq.with_suffix(".work")
        os.replace(rq, work)
        try:
            r = json.loads(work.read_text())
            run = Path(r["run"]) if os.path.isabs(r["run"]) else run_root / r["run"]
            cells = r.get("cells") or Path(r["cells_file"]).read_text().split()
            res = gpu_shard(run, cells=cells, session=ses, redo_failed=bool(r.get("redo_failed")),
                            cache=cache, workers=workers)
        except SystemExit:
            work.with_suffix(".done").write_text(json.dumps({"error": "HIP fault (worker exits 86)"}))
            raise
        except Exception as e:  # noqa: BLE001 -- a bad request must not kill the worker
            res = {"error": repr(e)[-1500:]}
        work.with_suffix(".done").write_text(json.dumps(res, default=str))
        print(f"[serve] {rq.name}: {res}", flush=True)
    return 0


def gpu_shard(run: Path, shard: int = 0, nshards: int = 1, cache: Optional[Path] = None, workers: int = 16,
              cells: Optional[Sequence[str]] = None, redo_failed: bool = False, redo_noisy: bool = False,
              session: Optional[GpuSession] = None) -> dict:
    """Verify and time this shard's cells on the visible GPU (resumable: done cells are skipped; ``redo_noisy``:
    also the done cells whose twin gate failed)."""
    pl = json.loads((run / "plan.json").read_text())
    man = pl["manifest"]
    led = Ledger(run, "gpu", f"{shard}of{nshards}")
    st = Ledger.state(run)
    all_cells = {c["id"]: c for c in pl["cells"]}
    ids = list(cells) if cells else _shard(list(all_cells), shard, nshards)
    if cells:  # an explicit list is run as given (re-measurement), done or not
        todo = [c for c in ids if c in all_cells]
    else:
        todo = [c for c in ids if (st.get(f"cell:{c}") != "done" or (redo_noisy and _noisy(run, c))) and
                (redo_failed or st.get(f"cell:{c}") not in ("verify_failed", "build_missing", "error"))]
    print(f"[gpu] shard {shard}/{nshards}: {len(todo)} of {len(ids)} cells to do", flush=True)
    rng = random.Random(man["seed"] + shard)
    if _fake():
        for cid in todo:
            res = {"cell": all_cells[cid], **_fake_cell(all_cells[cid], rng), "bdf": "fake", "node": "fake"}
            (run / "cells" / f"{_sha(cid)}.json").write_text(json.dumps(res, indent=1, default=str))
            led.add(item=f"cell:{cid}", status="done")
        return {"done": len(todo)}
    ses = session or GpuSession(cache or (run / "cache"), int(man["seed"]), workers)
    rt, g, L, TU, clk, bdf, node = ses.rt, ses.g, ses.L, ses.TU, ses.clk, ses.bdf, ses.node
    data = ses.data_for(cache or (run / "cache"), int(man["seed"]), workers)
    led.add(item=f"gpu:{shard}", status="start", bdf=bdf, node=node)
    n_done = 0
    for cid in todo:
        cell = all_cells[cid]
        missing = [b for b in [cell["build"]] if obj_path(run, b) is None]
        if missing:
            led.add(item=f"cell:{cid}", status="build_missing", builds=missing)
            continue
        t0 = time.time()
        try:
            res = _run_cell(run, rt, g, L, TU, clk, data, cell, man, rng)
        except Exception as e:  # noqa: BLE001 -- a fault ends this process (HIP context poisoned): record and stop
            led.add(item=f"cell:{cid}", status="error", arm=CURRENT_ARM[0], error=repr(e)[-2000:])
            print(f"[gpu] {cid}: ERROR in arm {CURRENT_ARM[0]}: {e!r}", flush=True)
            if "hipError" in repr(e) or "HipError" in repr(e):
                raise SystemExit(86)
            continue
        res.update({"bdf": bdf, "node": node, "job": os.environ.get("SLURM_JOB_ID"), "s": round(time.time() - t0, 1)})
        (run / "cells" / f"{_sha(cid)}.json").write_text(json.dumps(res, indent=1, default=str))
        status = "done" if res["verify"]["ok"] else "verify_failed"
        led.add(item=f"cell:{cid}", status=status, s=res["s"])
        n_done += 1
        tw = res.get("timing", {}).get("twin_ratio_warm", {})
        print(f"[gpu] {cid}: {status} warm {res.get('timing', {}).get('cand_warm_tflops')} twin {tw.get('median')}",
              flush=True)
    led.add(item=f"gpu:{shard}", status="end")
    return {"done": n_done}


class _PrePair:
    """am lane: a GEMM launch preceded by the transpose that fills its A workspace (``_PreRuntime.launch``)."""

    def __init__(self, fn, tr_fn, tr_grid, tr_block, tr_ka, ws: int, ws_bytes: int):
        self.fn, self.tr = fn, (tr_fn, tr_grid, tr_block, tr_ka)
        self.ws, self.ws_bytes, self.clobber = ws, ws_bytes, False


class _PreRuntime:
    """The HIP runtime with ``launch`` of a ``_PrePair`` = the transpose, then the GEMM (verification clobbers the
    workspace first, so a transpose that writes nothing cannot pass on the host-packed canonical image)."""

    def __init__(self, rt):
        self._rt = rt

    def __getattr__(self, name):
        return getattr(self._rt, name)

    def launch(self, fn, grid, block, ka):
        if isinstance(fn, _PrePair):
            if fn.clobber:
                self._rt.memset(fn.ws, 0xFF, fn.ws_bytes)
            self._rt.launch(*fn.tr)
            return self._rt.launch(fn.fn, grid, block, ka)
        return self._rt.launch(fn, grid, block, ka)


_TR_OBJS: Dict[str, bytes] = {}


def _tr_module(rt, run: Path, ts: Mapping[str, object]):
    """(module, function, spec) of a transpose spec: built with LLVM (comgr) in this process, cached in ``run/objs``."""
    from kernels.gfx950.mxfp6_gemm import transpose as TR
    from rocke.helpers.compile import compile_kernel

    spec = replace(TR.TrSpec(), **dict(ts))
    nm = TR.kernel_name(spec)
    p = run / "objs" / f"tr-{nm}.hsaco"
    if nm not in _TR_OBJS:
        if not p.exists():
            hs = compile_kernel(TR.build_transpose(spec), arch="gfx950", backend="python").hsaco
            tmp = p.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(hs)
            os.replace(tmp, p)
        _TR_OBJS[nm] = p.read_bytes()
    m = rt.load_module(_TR_OBJS[nm])
    return m, m.get_function(nm), spec


def _run_cell(run: Path, rt, g, L, TU, clk, data: Data, cell: Mapping, man: Mapping, rng: random.Random) -> dict:
    spec = replace(g.Mxfp6GemmSpec(), **cell["spec"])
    if cell.get("pre"):  # am lane: transpose (M-contiguous A -> the GEMM's canonical A buffer), then the ak GEMM
        rt = _PreRuntime(rt)
    gc = GpuCell(rt, L, g, data, cell, man)
    arms: Dict[str, dict] = {}
    mods = []

    def load(name: str, bid: str, over: Mapping) -> None:
        meta = obj_meta(run, bid)
        sp = replace(g.Mxfp6GemmSpec(), **dict(over))
        m = rt.load_module(obj_path(run, bid).read_bytes())
        mods.append(m)
        p = cell["problem"]
        if getattr(sp, "runtime_shape", False):
            grid, block = g.grid_for(sp, int(p["m"]), int(p["n"]), int(p["batch"]))
        else:
            grid, block = g.grid_of(sp)
        arms[name] = {"fn": m.get_function(meta["name"]), "grid": grid, "block": block, "spec": sp, "meta": meta}

    builds = json.loads((run / "plan.json").read_text())["builds"]
    load("cand", cell["build"], cell["spec"])
    load("twin", cell["build"], cell["spec"])  # A-vs-A: second module load of the same code object
    probe_memo = MEMO["probe"].get(cell.get("probe")) if man.get("probe_once") else None
    if cell.get("probe") and probe_memo is None and obj_path(run, cell["probe"]) is not None:
        load("probe", cell["probe"], builds[cell["probe"]]["spec"])
    mut_memo = MEMO["mut"].get(cell["build"]) if man.get("mutants_once") else None
    for mu, bid in cell.get("mutants", {}).items():
        if mut_memo is None and obj_path(run, bid) is not None:
            load(f"mut_{mu}", bid, builds[bid]["spec"])
    n_src = int(man["n_src"])

    def ka(name: str, dist: str, set_idx: int = 0, nsrc: int = n_src) -> bytes:
        x = arms[name]
        buf = gc.inputs(x["spec"], dist, nsrc, set_idx)
        ptrs = buf["ptrs"]
        if getattr(x["spec"], "splitk", "none") != "none":  # split-K: per-arm workspace + counters (zeroed once)
            ptrs = {**ptrs, **gc.sk_buffers(name, x["spec"])}
        return kernargs(g, x["spec"], ptrs, cell["problem"], buf["strides"])

    if cell.get("pre"):  # am lane: every GEMM arm is a (transpose, GEMM) pair per input set; + the transpose mutant
        from kernels.gfx950.mxfp6_gemm import transpose as TR

        pp = cell["problem"]
        tmods = {}
        for key, tsd in (("tr", cell["pre"]), ("tr_mut", {**cell["pre"], "mutant": "noswz"})):
            tm_, tf_, ts_ = _tr_module(rt, run, tsd)
            mods.append(tm_)
            tmods[key] = (tf_, ts_)
        pairs: Dict[tuple, _PrePair] = {}

        def fn_of(name: str, dist: str, set_idx: int = 0, nsrc: int = n_src, trk: str = "tr", clobber=False):
            key = (name, dist, set_idx, nsrc, trk)
            if key not in pairs:
                x = arms[name]
                buf = gc.inputs(x["spec"], dist, nsrc, set_idx)
                mc = gc.inputs(replace(x["spec"], a_layout="mcontig"), dist, nsrc, set_idx)
                src = mc["ptrs"]["a"]
                tf_, ts_ = tmods[trk]
                grid_t, block_t = TR.grid(ts_, int(pp["m"]), int(pp["k"]), int(pp["batch"]))
                ws = buf["ptrs"]["a"]
                # explicit entry strides: the M-contiguous input's (M % 16 != 0: ceil16(M) codes per K row) and the
                # GEMM's canonical A
                kt = TR.kargs(ts_, src, ws, int(pp["m"]), int(pp["k"]), int(mc["strides"]["stride_a"]),
                              int(buf["strides"]["stride_a"]))
                pairs[key] = _PrePair(x["fn"], tf_, grid_t, block_t, kt, ws,
                                      int(buf["strides"]["stride_a"]) * int(pp["batch"]))
            pairs[key].clobber = clobber
            return pairs[key]
    else:
        def fn_of(name: str, dist: str, set_idx: int = 0, nsrc: int = n_src, trk: str = "tr", clobber=False):
            return arms[name]["fn"]

    # ---------------- verify (distinct sources cycled over entries; determinism; mutants must fail)
    ver: Dict[str, dict] = {}
    vdists = list(man["verify_dists"])
    if getattr(spec, "splitk", "none") != "none":
        # split-K with s >= 3 split tiles: + the order set (exact partials whose sum depends on the fold order), the
        # only data on which a fix-up that reorders the partials (sk_order / sk_arrival) can fail -- integer data
        # sums exactly in any order and float data reorders far below the tolerance
        from kernels.gfx950.mxfp6_gemm import splitk as _SK

        _p = _SK.plan(spec, gc.m, gc.n, gc.k, int(cell["problem"].get("batch", 1)))
        if _p["D"] < _p["T"] and _p["s"] >= 3:
            vdists.append(f"order{_p['s']}")
    for dist in vdists:
        CURRENT_ARM[0] = "cand"
        outs = [gc.run_once(fn_of("cand", dist, clobber=True), arms["cand"]["grid"], arms["cand"]["block"],
                            ka("cand", dist)) for _ in range(2)]
        CURRENT_ARM[0] = "twin"
        chk = _check_cell(L, data, outs[0], dist, spec.elem_fmt, gc.m, gc.n, gc.k, n_src)
        chk["deterministic"] = bool(np.array_equal(outs[0], outs[1]))
        tw = gc.run_once(fn_of("twin", dist, clobber=True), arms["twin"]["grid"], arms["twin"]["block"],
                         ka("twin", dist))
        chk["twin_bitwise"] = bool(np.array_equal(outs[0], tw))
        ver[dist] = chk
    muts = {k: {**v, "reused": True} for k, v in (mut_memo or {}).items()}
    for name in [a for a in arms if a.startswith("mut_")]:
        fails = False
        try:  # the kernel refuses a mutant that does not apply to this problem (batch_swap at batch < 2)
            ka(name, man["verify_dists"][0])
        except ValueError as e:
            muts[name] = {"failed_as_required": None, "not_applicable": str(e)[:200]}
            continue
        for dist in vdists:
            if dist == "int" and "scale_sel" in name:
                continue  # all scales 2^0 on the integer set: op_sel cannot show
            CURRENT_ARM[0] = name
            o = gc.run_once(fn_of(name, dist, clobber=True), arms[name]["grid"], arms[name]["block"], ka(name, dist))
            if not _check_cell(L, data, o, dist, spec.elem_fmt, gc.m, gc.n, gc.k, n_src)["pass"]:
                fails = True
                break
        muts[name] = {"failed_as_required": fails}
    if cell.get("pre"):  # the transpose's own must-fail mutant inside the pair (with the candidate GEMM)
        fails = False
        for dist in man["verify_dists"]:
            CURRENT_ARM[0] = "pre_noswz"
            o = gc.run_once(fn_of("cand", dist, trk="tr_mut", clobber=True), arms["cand"]["grid"],
                            arms["cand"]["block"], ka("cand", dist))
            if not _check_cell(L, data, o, dist, spec.elem_fmt, gc.m, gc.n, gc.k, n_src)["pass"]:
                fails = True
                break
        muts["pre_noswz"] = {"failed_as_required": fails}
    ok = all(v["pass"] and v["deterministic"] and v["twin_bitwise"] for v in ver.values()) and \
        all(m["failed_as_required"] for m in muts.values() if not m.get("not_applicable"))
    if man.get("mutants_once") and mut_memo is None and muts:
        MEMO["mut"][cell["build"]] = {k: v for k, v in muts.items()}
    out = {"cell": dict(cell), "verify": {"ok": ok, "dists": ver, "mutants": muts},
           "build": {k: arms["cand"]["meta"].get(k) for k in ("name", "compiler", "notes", "loop", "rust_levers",
                                                              "rust_tool", "rust_tool_sha256", "ir_sha256")}}
    if "probe" in arms:
        out["probe_build"] = {k: arms["probe"]["meta"].get(k) for k in ("name", "notes", "loop")}
    if not ok:
        for m in mods:
            m.unload()
        gc.free()
        return out
    # ---------------- time (rotated replicated input sets; cand / twin / probe interleaved)
    p = cell["problem"]
    rot = max(1, int(p.get("rotate", 1)))
    timed = [a for a in ("cand", "twin", "probe") if a in arms]
    timing: Dict[str, dict] = {"rotate": rot, "dists": {}}
    CURRENT_ARM[0] = "timing:" + "/".join(timed)
    for dist in man["time_dists"]:
        seqs = {a: [(fn_of(a, dist, s, 1) if a != "probe" else arms[a]["fn"], arms[a]["grid"], arms[a]["block"],
                     ka(a, dist, s, 1)) for s in range(rot)] for a in timed}
        res = {a: {"cold": [], "warm": []} for a in timed}
        for _ in range(int(man["cold_rounds"])):  # cold-start (post-idle): gap, 1 warmup + 3 timed
            order = list(timed)
            rng.shuffle(order)
            for a in order:
                s = seqs[a]
                rt.wait_stream(0)
                time.sleep(float(man["gap_ms"]) / 1000.0)
                rt.launch(*s[0])
                e0, e1 = rt.event(), rt.event()
                e0.record()
                for i in range(3):
                    rt.launch(*s[(i + 1) % rot])
                e1.record()
                e1.synchronize()
                res[a]["cold"].append(3 * gc.flop / (e0.elapsed_to(e1) * 1e9))
                e0.destroy()
                e1.destroy()

        def warm_windows(n: int) -> None:  # warm (steady-state) windows, arms shuffled, SCLK / power sampling
            for _ in range(n):
                order = list(timed)
                rng.shuffle(order)
                for a in order:
                    s = seqs[a]
                    rt.wait_stream(0)
                    t0 = time.time()
                    for i in range(5):
                        rt.launch(*s[i % rot])
                    rt.wait_stream(0)
                    est_ms = max(0.01, (time.time() - t0) * 1000.0 / 5)
                    iters = max(50, int(float(man["warm_s"]) * 1000.0 / est_ms))
                    for i in range(20):
                        rt.launch(*s[i % rot])
                    clk.start()
                    e0, e1 = rt.event(), rt.event()
                    e0.record()
                    for i in range(iters):
                        rt.launch(*s[i % rot])
                    e1.record()
                    e1.synchronize()
                    ms = e0.elapsed_to(e1) / iters
                    e0.destroy()
                    e1.destroy()
                    cs = clk.stop()
                    w = cs.get("watts_med")
                    tf = gc.flop / (ms * 1e9)
                    res[a]["warm"].append({"tflops": tf, "ms": ms, "iters": iters, "mj": w * ms if w else None,
                                           "pj_per_flop": w / tf if w else None, **cs})

        def twin_ok() -> bool:
            ci = TU.bootstrap_ci([x["tflops"] for x in res["cand"]["warm"]], [x["tflops"] for x in res["twin"]["warm"]])
            return (ci["median"] is not None and abs(ci["median"] - 1.0) <= TWIN_TOL
                    and (ci["hi"] - ci["lo"]) <= 2 * TWIN_TOL)

        n_win, retimes = int(man["warm_windows"]), 0
        while True:
            warm_windows(n_win)
            if twin_ok() or retimes >= TWIN_RETRIES:
                break
            retimes += 1  # a throttling / neighbour episode spoiled windows: more interleaved windows, then re-judge
            n_win = int(man["warm_windows"])
        d_retimes = retimes
        d_twin_gate = twin_ok()
        d = {"arms": res, "retimes": d_retimes, "twin_gate": d_twin_gate}
        for a in timed:
            d[f"{a}_warm_tflops"] = _median([x["tflops"] for x in res[a]["warm"]])
            d[f"{a}_cold_tflops"] = _median(res[a]["cold"])
            d[f"{a}_warm_mj"] = _median([x["mj"] for x in res[a]["warm"]])
            d[f"{a}_warm_sclk"] = _median([x.get("sclk_med") for x in res[a]["warm"]])
            d[f"{a}_warm_watts"] = _median([x.get("watts_med") for x in res[a]["warm"]])
        d["twin_ratio_warm"] = TU.bootstrap_ci([x["tflops"] for x in res["cand"]["warm"]],
                                               [x["tflops"] for x in res["twin"]["warm"]])
        d["twin_ratio_cold"] = TU.bootstrap_ci(res["cand"]["cold"], res["twin"]["cold"])
        if "probe" in res:
            d["of_probe_warm"] = TU.bootstrap_ci([x["tflops"] for x in res["cand"]["warm"]],
                                                 [x["tflops"] for x in res["probe"]["warm"]])
        timing["dists"][dist] = d
    head = timing["dists"].get("gauss") or next(iter(timing["dists"].values()))
    timing.update({k: v for k, v in head.items() if k != "arms"})
    if man.get("probe_once") and cell.get("probe"):
        if probe_memo is not None:
            timing["probe_warm_tflops"], timing["probe_reused"] = probe_memo, True
        elif timing.get("probe_warm_tflops"):
            MEMO["probe"][cell["probe"]] = timing["probe_warm_tflops"]
    out["timing"] = timing
    for m in mods:
        m.unload()
    gc.free()
    return out


# ================================================================== table


def load_ceilings(path: Optional[str]) -> dict:
    """Ceiling inputs (workspace JSON, not in the repo): ``{"spec_tflops": the product's dense MXFP6 spec,
    "families": {family_key: {"gauss_e2m3": TF/s,
    "gauss_e3m2": .., "zero": .., "lds_gauss_e2m3": ..}}, "bw": {"lds_b64": B/clk/CU, "lds_b128": ..,
    "l2": B/clk/CU, "mall": B/s, "hbm": B/s}, "node": .., "bdf": ..}``."""
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def ceiling_of(cell: Mapping, res: Mapping, ceil: Mapping, same_job: Optional[Mapping] = None) -> dict:
    """The cell's ceiling: min(MFMA-only ceiling, bandwidth roofline at the cell's warm clock). MFMA-only ceiling,
    first available of: the kernel's own probe="mfma_only" arm (same rounds), the C++ family probe run by the same
    job on the same GPU (``ceil/``), the family table (``ceil["families"]``, reference node)."""
    notes = (res.get("build") or {}).get("notes") or {}
    fam, w, occ = cell_family(cell, notes)
    out: Dict[str, object] = {"family": fam, "occ": occ}
    timing = res.get("timing") or {}
    fmt = cell["fmt"]
    mf = ((ceil.get("families") or {}).get(fam) or {}).get(f"gauss_{fmt}")
    out["mfma_table"] = mf
    out["mfma_probe"] = timing.get("probe_warm_tflops")
    sj = ((same_job or {}).get((str(res.get("job")), str(res.get("bdf")).lower())) or {}).get(f"{fam}/gauss_{fmt}")
    out["mfma_same_job"] = sj
    # the highest same-job MFMA-only rate is the denominator (the kernel's own mfma_only body carries its scale
    # loads / address math, the C++ family probe does not); the reference-node family table only as a fallback
    # the reference-node family table, scaled to this GPU by the same job's reference-family probe (board offset)
    ref = "16x16x128_t8x8_k2_w4_o1"
    sj_ref = ((same_job or {}).get((str(res.get("job")), str(res.get("bdf")).lower())) or {}).get(f"{ref}/gauss_{fmt}")
    t_ref = ((ceil.get("families") or {}).get(ref) or {}).get(f"gauss_{fmt}")
    mf_gpu = mf * sj_ref / t_ref if mf and sj_ref and t_ref else None
    out["mfma_table_gpu"] = mf_gpu
    # the denominator is the highest MFMA-only rate measured for this family on this GPU: a probe that under-reads
    # (e.g. a spilling 8-wave C++ build) never lowers it
    # a C++ family probe far below the same GPU's reference family is a probe artifact (e.g. an 8-wave build that
    # spills at 2 waves / SIMD): saturated MFMA-only ceilings are flat within ~10 % at the cap ("Ceilings" finding 1)
    if sj and sj_ref and sj < 0.7 * sj_ref:
        out["probe_rejected"] = sj
        sj = None
    same = [(v, k) for v, k in ((out["mfma_probe"], "kernel probe"), (sj, "same-job C++ probe"),
                                (mf_gpu, "family table x same-GPU ref")) if v]
    if not same and sj_ref:
        same = [(sj_ref, "same-GPU reference family (flat ceiling)")]
    mfma, src = max(same) if same else (mf, "family table" if mf else None)
    out["mfma_source"] = src
    sclk = timing.get("cand_warm_sclk")
    bw = ceil.get("bw") or {}
    prob = T.Problem(**{k: v for k, v in cell["problem"].items() if k in ("set", "tile", "batch", "m", "n", "k",
                                                                          "tiles", "rotate", "note")})
    rf = T.roofline(w, {"lds": bw.get("lds_b128", 0.0), "l2": bw.get("l2", 0.0), "mall": 0.0, "hbm": 0.0}, occ, prob)
    roofs = {}
    if sclk and bw:
        hz = T.N_CU * sclk * 1e6 * T.FLOP_CLK_CU / 1e12  # TF/s at 100 % MFMA issue at this clock
        roofs["lds_b128"] = rf["lds"] * hz
        roofs["lds_b64"] = bw.get("lds_b64", 0.0) * w.ai_lds / T.FLOP_CLK_CU * hz
        roofs["l2"] = rf["l2"] * hz
        if cell.get("family") == "am":  # M-contiguous A: 16-byte DMA slots carry 12 code bytes (+1/3 A traffic)
            tile = T.Tile.parse(cell["tile"])
            extra = tile.tile_m * tile.du * T.B_CODE / 3.0
            roofs["l2"] *= w.glob_bytes / (w.glob_bytes + extra)
        if bw.get("mall"):
            roofs["fabric"] = bw["mall"] * rf["ai_fabric"] / 1e12
        if bw.get("hbm"):
            roofs["hbm"] = bw["hbm"] * rf["ai_hbm"] / 1e12
    out["roofs"] = roofs
    cands = [v for v in [mfma] + [roofs.get(k) for k in ("lds_b128", "l2", "fabric", "hbm")] if v]
    out["ceiling"] = min(cands) if cands else None
    out["bound"] = (min(((v, k) for k, v in [("mfma", mfma)] + [(k, roofs.get(k)) for k in
                                                                ("lds_b128", "l2", "fabric", "hbm")] if v))[1]
                    if cands else None)
    return out


def table(run: Path, ceilings: Optional[str] = None, fmt: str = "md") -> str:
    ceil = load_ceilings(ceilings)
    sj = same_job_ceilings(run)
    rows = []
    for p in sorted((run / "cells").glob("*.json")):
        r = json.loads(p.read_text())
        cell = r["cell"]
        tm = r.get("timing") or {}
        c = ceiling_of(cell, r, ceil, sj) if tm else {}
        warm = tm.get("cand_warm_tflops")
        tw = tm.get("twin_ratio_warm") or {}
        rows.append({
            "config": cell["config"], "tile": cell["tile"], "family": cell["family"], "fmt": cell["fmt"],
            "problem": f"{cell['problem']['set']}:{cell['problem']['batch']}x{cell['problem']['m']}x"
                       f"{cell['problem']['n']}x{cell['problem']['k']}",
            "verify": "ok" if r["verify"].get("ok") else "FAIL",
            "warm": warm, "cold": tm.get("cand_cold_tflops"), "mj": tm.get("cand_warm_mj"),
            "pj_flop": (tm.get("cand_warm_watts") / warm) if warm and tm.get("cand_warm_watts") else None,
            "sclk": tm.get("cand_warm_sclk"), "twin": tw.get("median"), "twin_ci": (tw.get("lo"), tw.get("hi")),
            "twin_gate": tm.get("twin_gate"), "retimes": tm.get("retimes"),
            "ceiling": c.get("ceiling"), "bound": c.get("bound"), "mfma_source": c.get("mfma_source"),
            "pct_ceiling": 100.0 * warm / c["ceiling"] if warm and c.get("ceiling") else None,
            "ceil_suspect": bool(warm and c.get("ceiling") and warm > 1.05 * c["ceiling"]),
            "pct_spec": 100.0 * warm / ceil["spec_tflops"] if warm and ceil.get("spec_tflops") else None,
            "occ": c.get("occ"), "node": r.get("node"), "bdf": r.get("bdf"), "job": r.get("job"),
        })
    if fmt == "json":
        return json.dumps(rows, indent=1, default=str)

    def f(x, d=1):
        return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))

    hdr = ("| config | fmt | problem | verify | warm TF/s | cold TF/s | mJ | pJ/FLOP | SCLK | twin [CI] | ceiling (bound) "
           "| ceil src | % ceil | % spec | occ | node/bdf/job |")
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in rows:
        ci = r["twin_ci"]
        lines.append(f"| {r['config']} | {r['fmt']} | {r['problem']} | {r['verify']} | {f(r['warm'])} | {f(r['cold'])} | "
                     f"{f(r['mj'])} | {f(r['pj_flop'], 3)} | {f(r['sclk'], 0)} | {'NOISY ' if r['twin_gate'] is False else ''}{f(r['twin'], 4)} "
                     f"[{f(ci[0], 4)}, {f(ci[1], 4)}] | {f(r['ceiling'])} ({r['bound'] or '-'}) | {r['mfma_source'] or '-'} | "
                     f"{f(r['pct_ceiling'])} | "
                     f"{f(r['pct_spec'])} | {r['occ'] or '-'} | {r['node']}/{r['bdf']}/{r['job']} |")
    return "\n".join(lines)


def cell_family(cell: Mapping, notes: Mapping) -> Tuple[str, T.Work, int]:
    """Probe family / work model of the kernel the cell actually ran: its work-group tile (``tile_m`` / ``tile_n`` of
    the spec, which a multi-tile WG (WG_MULT) makes larger than the library tile) and its waves."""
    lt = T.Tile.parse(cell["tile"])
    sp = cell["spec"]
    tile = T.Tile(int(sp.get("tile_n", lt.tile_n)), int(sp.get("tile_m", lt.tile_m)), lt.du)
    lay = T.Layout(str(sp.get("atom", "16x16x128")), int(sp.get("waves_m", 2)), int(sp.get("waves_n", 2)))
    if not T.layout_ok(tile, lay):
        lay = T.default_layout(tile)
    w = T.work(tile, lay)
    occ = occupancy_of(notes, lay.waves) or max(w.occ_est, 1)
    return T.family_key(w, occ), w, occ


def families(run: Path, job: Optional[str] = None, bdf: Optional[str] = None) -> List[str]:
    """Probe families (``mfma_tile.cpp`` build parameters) of the timed cells (of one job): lines
    ``family atom tm tn ks kd waves occ lds``."""
    out = {}
    for p in sorted((run / "cells").glob("*.json")):
        r = json.loads(p.read_text())
        if not r.get("timing") or (job and str(r.get("job")) != str(job)) or                 (bdf and str(r.get("bdf")).lower() != bdf.lower()):
            continue
        fam, w, occ = cell_family(r["cell"], (r.get("build") or {}).get("notes") or {})
        atom = 16 if w.atom == "16x16x128" else 32
        kd = 2 if w.acc_regs + 2 * (w.tm + w.tn) * 7 <= 440 and w.ks % 2 == 0 else 1
        lds = (min(w.lds_est, T.LDS_BYTES // occ) // 1024) * 1024
        out[fam] = f"{fam} {atom} {w.tm} {w.tn} {w.ks} {kd} {w.waves} {occ} {lds}"
    return [out[k] for k in sorted(out)]


def same_job_ceilings(run: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """``ceil/*.jsonl`` rows written by the GPU jobs' probe step: (job, bdf) -> {family/variant: warm TF/s}."""
    out: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for p in sorted((run / "ceil").glob("*.jsonl")) if (run / "ceil").exists() else []:
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r or r.get("mode") != "warm":
                continue
            var = "zero" if r.get("data") == 0 else ("gauss_e3m2" if r.get("fmt") == 3 else "gauss_e2m3")
            key = (str(r.get("job")), str(r.get("bdf")).lower())
            out.setdefault(key, {}).setdefault(f"{r['family']}/{var}", []).append(r["tflops"])
    return {k: {f: statistics.median(v) for f, v in d.items()} for k, d in out.items()}


def status(run: Path) -> dict:
    pl = json.loads((run / "plan.json").read_text())
    st = Ledger.state(run)
    cnt: Dict[str, Dict[str, int]] = {"build": {}, "cell": {}}
    for bid in pl["builds"]:
        s = st.get(f"build:{bid}", "pending")
        cnt["build"][s] = cnt["build"].get(s, 0) + 1
    for c in pl["cells"]:
        s = st.get(f"cell:{c['id']}", "pending")
        cnt["cell"][s] = cnt["cell"].get(s, 0) + 1
    return {"builds": len(pl["builds"]), "cells": len(pl["cells"]), "refused": len(pl["refused"]), **cnt}


# ================================================================== CLI


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--manifest", required=True)
    a.add_argument("--run", required=True)
    a = sub.add_parser("compile")
    a.add_argument("--run", required=True)
    a.add_argument("--shard", default="0/1", help="i/n")
    a.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a.add_argument("--redo-failed", action="store_true")
    a = sub.add_parser("gpu")
    a.add_argument("--run", required=True)
    a.add_argument("--shard", default="0/1", help="i/n")
    a.add_argument("--cache", default=None, help="operand / reference cache (node-local)")
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--cells", default=None, help="file with one cell id per line, or a comma list (run as given, "
                   "done or not; default: the shard's pending cells)")
    a.add_argument("--redo-failed", action="store_true")
    a.add_argument("--redo-noisy", action="store_true", help="re-time done cells whose A-vs-A twin gate failed")
    a = sub.add_parser("table")
    a.add_argument("--run", required=True)
    a.add_argument("--ceilings", default=None)
    a.add_argument("--format", choices=("md", "json"), default="md")
    a = sub.add_parser("status")
    a.add_argument("--run", required=True)
    a = sub.add_parser("serve", help="persistent GPU worker: run requests dropped into --inbox (see serve())")
    a.add_argument("--run", required=True, help="run root (requests may name runs relative to it)")
    a.add_argument("--inbox", required=True)
    a.add_argument("--cache", required=True)
    a.add_argument("--workers", type=int, default=16)
    a = sub.add_parser("buildd", help="long-lived build service on a CPU node (see buildd())")
    a.add_argument("--queue", required=True)
    a.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a.add_argument("--current", default=None, help="the lane's CURRENT pointer (restart when it moves)")
    a.add_argument("--idle-exit", type=float, default=0.0, help="exit after this many idle seconds (0: never)")
    a = sub.add_parser("families")
    a.add_argument("--run", required=True)
    a.add_argument("--job", default=None)
    a.add_argument("--bdf", default=None)
    args = p.parse_args(argv)
    run = Path(getattr(args, "run", None) or ".")
    if args.cmd == "plan":
        out = plan(load_manifest(args.manifest), run)
        print(json.dumps({"cells": len(out["cells"]), "builds": len(out["builds"]), "refused": out["refused"]},
                         indent=1))
        return 0
    if args.cmd in ("compile", "gpu"):
        i, n = (int(x) for x in args.shard.split("/"))
        if args.cmd == "compile":
            r = compile_shard(run, i, n, args.jobs, args.redo_failed)
            return 0 if not r["failed"] else 1
        cl = None
        if args.cells:
            cl = (Path(args.cells).read_text().splitlines() if os.path.exists(args.cells) else args.cells.split(","))
            cl = _shard([c.strip() for c in cl if c.strip()], i, n)
        r = gpu_shard(run, i, n, Path(args.cache) if args.cache else None, args.workers, cl, args.redo_failed,
                      args.redo_noisy)
        return 0
    if args.cmd == "table":
        print(table(run, args.ceilings, args.format))
        return 0
    if args.cmd == "serve":
        return serve(run, Path(args.inbox), Path(args.cache), args.workers)
    if args.cmd == "buildd":
        return buildd(Path(args.queue), args.jobs, Path(args.current) if args.current else None, args.idle_exit)
    if args.cmd == "families":
        print("\n".join(families(run, args.job, args.bdf)))
        return 0
    print(json.dumps(status(run), indent=1))
    return 0


def _no_gc():
    """The cyclic GC paused while an IR is built / serialized (2x faster; kernels.gfx950.mxfp6_gemm.gemm.no_gc)."""
    from kernels.gfx950.mxfp6_gemm.gemm import no_gc

    return no_gc()


if __name__ == "__main__":  # (last: `python -m` runs main() here, so every helper above must already be defined)
    raise SystemExit(main())
