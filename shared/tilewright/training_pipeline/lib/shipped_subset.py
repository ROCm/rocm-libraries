# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which bank kernels to ship next to a library pool: the candidate set the
engine ranks at runtime is the pool plus a subset S of its bank
(lib/library_pool.py), and a larger S raises the ceiling but also
the ranking time and the chance that the cascade keeps a worse key.

The deployed pick of a GEMM under S is simulated from per-GEMM score
matrices computed once over the whole universe (pool, then bank, in shipped
order), following tilewright's `rank_in_cell` for an MLREC_v3 model:

  keys       configs with equal `features.v3_candidate_key` under the
             launch model the engine groups the model's pool by
             (`ModelView.key_launch`: what `lib.evaluate.candidate_set`
             hands the engine when the model serves launch keys, else none)
             are one candidate (numbered by first appearance in the shipped
             config list); only a key's first member gives its launch plan,
             so its full score is the full score of its first present
             member;
  views      keys with equal `features.v3_view_key` (launch classes of one
             view key) share feasibility, whitelisting and the embedding
             term;
  tier       the keys that pass the feasibility rules (`lib.feasibility`,
             with non-temporal availability from pool + S), restricted to
             the cell's whitelisted keys when the model serves with a
             whitelist tier and any of them is present;
  cascade    a tier without a whitelist with more views than the model's
             `cascade_keys` keeps the keys of its best views by the
             embedding term alone (q . i / temperature, ties to the earlier
             view);
  ranking    kept keys by full score (non-finite: unscored), each key's
             members in config order, keys of equal score interleaved in
             config order;
  pick       the first ranked config the runtime runs, charged by
             `lib.evaluate.pick_cost` under the run's pick measurement,
             else the runtime's fallback pick.

The engine's full scores are exact: each config is ranked once as the first
member of its key, in candidate sets split by non-temporal hints so that
every feasible key is scored. The embedding term is recomputed in numpy from
the engine's raw features and the decoded weights (the bindings expose no
stage-1 score); `cascade_parity` checks it against `CandidateSet.rank`.

The subset is grown greedily, one bank config at a time, by the gain in the
summed log selection efficiency of the selection GEMMs (an unserved GEMM is
charged its fallback, else its slowest measurement); per-GEMM pick states
make each step touch only the GEMMs whose pick the chosen config can change.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import evaluate as ev
from . import feasibility
from . import features as fs
from . import hipblaslt_launch
from . import mlrec
from .hardware import DATATYPE_VALUE, DeviceHardware

DEFAULT_SIZES: Tuple[Optional[int], ...] = (0, 32, 64, 128, 256, 512, 1024, None)
BUDGETS_US: Tuple[float, ...] = (10.0, 20.0)
RECOMMEND_TOLERANCE = 0.002
LOSS = 0.99
BENCH_FIELDS = (
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "occupancy",
    "cache_hints_a",
    "cache_hints_b",
    "grvw_a",
    "grvw_b",
    "gwvw_d",
)
_NO_POS = np.iinfo(np.int64).max
_CHUNK = 1024


def size_label(b: Optional[int]) -> str:
    return "all" if b is None else str(int(b))


def parse_sizes(text: str) -> List[Optional[int]]:
    """'0,32,all' -> [0, 32, None]."""
    out: List[Optional[int]] = []
    for part in str(text).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "all":
            out.append(None)
            continue
        v = int(part)
        if v < 0:
            raise ValueError(f"negative size {v}")
        out.append(v)
    if not out:
        raise ValueError("no sizes")
    return out


# ── the universe of shippable configs ────────────────────────────────────────


@dataclass
class Universe:
    """Pool kernels (always shipped) then bank kernels, in shipped order.
    `key` is each config's candidate key; `members` the configs of every key
    in config order (a -1 padded table); `view` every key's view, and
    `view_first` every view's first key. `hashes` and `sigs` are per key, of
    its view key."""

    kernels: List[Dict[str, Any]]
    n_pool: int
    sids: np.ndarray
    key: np.ndarray
    members: np.ndarray
    shape: Dict[str, np.ndarray]
    nt_a: np.ndarray
    nt_b: np.ndarray
    hashes: np.ndarray
    sigs: List[Tuple[int, ...]]
    view: np.ndarray
    view_first: np.ndarray

    @property
    def n(self) -> int:
        return len(self.kernels)

    @property
    def n_keys(self) -> int:
        return int(self.members.shape[0])

    @property
    def n_views(self) -> int:
        return int(self.view_first.shape[0])

    def any_by_view(self, mask: np.ndarray) -> np.ndarray:
        """(rows, views) of a (rows, keys) mask: whether any key of the view
        is set."""
        order = np.argsort(self.view, kind="stable")
        starts = np.flatnonzero(np.r_[True, np.diff(self.view[order]) != 0])
        if not order.size:
            return np.zeros(mask.shape[:-1] + (0,), dtype=bool)
        return np.logical_or.reduceat(mask[..., order], starts, axis=-1)

    @property
    def bank(self) -> np.ndarray:
        return np.arange(self.n_pool, self.n)

    def present(self, bank_positions: Iterable[int] = ()) -> np.ndarray:
        """Shipped mask of the pool plus the given bank positions."""
        p = np.zeros(self.n, dtype=bool)
        p[: self.n_pool] = True
        idx = np.fromiter((int(i) for i in bank_positions), dtype=np.int64)
        if idx.size and (idx.min() < self.n_pool or idx.max() >= self.n):
            raise ValueError("a subset position is outside the bank")
        p[idx] = True
        return p

    def shipped(self, present: np.ndarray) -> List[Dict[str, Any]]:
        """The shipped config list (`present` in universe order)."""
        return [self.kernels[i] for i in np.flatnonzero(present)]

    def first_present(self, present: np.ndarray) -> np.ndarray:
        """Per key, its first present config (-1 when none is)."""
        m = self.members
        ok = (m >= 0) & present[np.where(m >= 0, m, 0)]
        first = ok.argmax(axis=1)
        return np.where(ok.any(axis=1), m[np.arange(m.shape[0]), first], -1)

    def flags(self, present: np.ndarray) -> Tuple[bool, bool]:
        """Non-temporal availability (A, B) of a shipped set."""
        return bool(self.nt_a[present].any()), bool(self.nt_b[present].any())


def build_universe(
    pool: Sequence[Mapping[str, Any]],
    bank: Sequence[Mapping[str, Any]],
    spec: fs.AttributeSpec,
    launch: Any = None,
) -> Universe:
    """The universe of `pool` then the kernels of `bank` not in the pool
    (by solution index), each in its given order, grouped as the engine
    groups them with launch model `launch` (`ModelView.key_launch`; None:
    by view key alone)."""
    kernels = [dict(k) for k in pool]
    seen = {int(k["sol_idx_global"]) for k in kernels}
    n_pool = len(kernels)
    for k in bank:
        sid = int(k["sol_idx_global"])
        if sid not in seen:
            seen.add(sid)
            kernels.append(dict(k))
    ids: Dict[Tuple[int, ...], int] = {}
    view_ids: Dict[Tuple[int, ...], int] = {}
    view: List[int] = []
    key = np.empty(len(kernels), dtype=np.int64)
    for i, k in enumerate(kernels):
        ck = fs.v3_candidate_key(
            fs.config_kwargs_from_row(k), k.get("attributes"), spec, launch
        )
        key[i] = ids.setdefault(ck, len(ids))
        if key[i] == len(view):
            view.append(view_ids.setdefault(ck[:-1], len(view_ids)))
    n_keys = len(ids)
    view_arr = np.array(view, dtype=np.int64)
    view_first = np.full(len(view_ids), -1, dtype=np.int64)
    for kk in range(n_keys - 1, -1, -1):
        view_first[view_arr[kk]] = kk
    counts = np.bincount(key, minlength=n_keys)
    members = np.full((n_keys, int(counts.max()) if n_keys else 0), -1, np.int64)
    fill = np.zeros(n_keys, dtype=np.int64)
    for i, kk in enumerate(key):
        members[kk, fill[kk]] = i
        fill[kk] += 1
    first = members[:, 0] if n_keys else np.zeros(0, np.int64)
    shape = {
        f: np.array([int(kernels[i][f]) for i in first], dtype=np.int64)
        for f in ev.CONFIG_FIELDS
    }
    hashes = np.array([fs.view_key_hash(ck[:-1]) for ck in ids], dtype=np.uint64)
    sigs = [
        tuple(int(kernels[i][f]) for f in fs.SIGNATURE_FIELDS) for i in first.tolist()
    ]
    nt = feasibility.NON_TEMPORAL
    return Universe(
        kernels=kernels,
        n_pool=n_pool,
        sids=np.array([int(k["sol_idx_global"]) for k in kernels], dtype=np.int64),
        key=key,
        members=members,
        shape=shape,
        nt_a=np.array([int(k["cache_hints_a"]) == nt for k in kernels]),
        nt_b=np.array([int(k["cache_hints_b"]) == nt for k in kernels]),
        hashes=hashes,
        sigs=sigs,
        view=view_arr,
        view_first=view_first,
    )


# ── the model ────────────────────────────────────────────────────────────────


@dataclass
class ModelView:
    """What the simulation needs of an MLREC_v3 model file. `launch` is
    the launch model `lib.evaluate.candidate_set` ranks it with, and
    `launch_keys` whether the model serves launch keys."""

    data: bytes
    hardware: DeviceHardware
    labels: List[str]
    cascade: int
    whitelist_tier: bool
    rules: Dict[str, bool]
    spec: fs.AttributeSpec
    whitelists: List[Tuple[set, set]]
    launch: Any = None
    launch_keys: bool = False

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def key_launch(self) -> Any:
        """The launch model whose launch keys the engine groups the pool
        by: `launch` when the model serves launch keys, else None."""
        return self.launch if self.launch_keys else None


def model_view(data: bytes, hardware: DeviceHardware, tw: Any) -> ModelView:
    """ValueError for a model that is not MLREC_v3 or whose attribute reads
    differ from the engine's."""
    parsed = mlrec.read_model(data)
    if parsed.get("version") != 3:
        raise ValueError("the subset simulation models MLREC_v3 ranking only")
    serving = parsed.get("serving") or {}
    unmodeled = [
        name
        for name, on in (
            (
                "the family-wise non-temporal rule",
                serving.get("non_temporal_family_rule"),
            ),
            ("the trained-key gate", serving.get("trained_key_gate")),
            ("the alignment veto", serving.get("alignment_veto")),
            ("the drift fallback", serving.get("drift_keep_ppm") is not None),
        )
        if on
    ]
    if unmodeled:
        raise ValueError(
            f"the subset simulation does not model {' or '.join(unmodeled)}, which "
            f"this model serves with"
        )
    model = ev.load_model_bytes(data, tw)
    info = tw.describe(model)
    spec = fs.parse_attribute_spec([a.to_dict() for a in parsed["attribute_spec"]])
    reads = [n for n, _ in fs.attribute_reads(spec)]
    if reads != list(tw.attribute_names(model)):
        raise ValueError(
            f"attribute reads {reads} differ from the engine's "
            f"{list(tw.attribute_names(model))}"
        )
    labels = [str(tw.cell_label(model, i)) for i in range(int(info.n_cells))]
    by_label = {c["label"]: c for c in parsed["cells"]}
    whitelists = []
    for lab in labels:
        c = by_label[lab]
        whitelists.append(
            (
                {tuple(int(v) for v in s) for s in c.get("smart_k_signatures") or []},
                {int(h) for h in c.get("view_key_hashes") or []},
            )
        )
    return ModelView(
        data=bytes(data),
        hardware=hardware,
        labels=labels,
        cascade=int(info.cascade_keys),
        whitelist_tier=bool(info.whitelist_tier),
        rules=feasibility.soft_rules(parsed.get("serving") or {}, 3),
        spec=spec,
        whitelists=whitelists,
        launch=hipblaslt_launch.HipblasltLaunch(str(info.arch), hardware.num_xcd),
        launch_keys=bool(info.launch_keys),
    )


def whitelist_masks(mv: ModelView, u: Universe) -> Tuple[np.ndarray, np.ndarray]:
    """(per cell: has a whitelist tier, per cell x key: whitelisted)."""
    n_cells = len(mv.labels)
    has = np.zeros(n_cells, dtype=bool)
    wl = np.zeros((n_cells, u.n_keys), dtype=bool)
    if not mv.whitelist_tier:
        return has, wl
    hashes = [int(h) for h in u.hashes]
    for ci, (sigs, vks) in enumerate(mv.whitelists):
        if not sigs and not vks:
            continue
        has[ci] = True
        wl[ci] = [s in sigs or h in vks for s, h in zip(u.sigs, hashes)]
    return has, wl


# ── engine scores ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoreJob:
    problem: Dict[str, Any]
    context: fs.ExecutionContext


_W: Dict[str, Any] = {}


def _whiten(x, mean, std, lo, hi):
    if lo is not None:
        x = np.minimum(np.maximum(x, lo), hi)
    scale = np.where(std < np.float32(1e-6), np.float32(1.0), std)
    return (x - mean) / scale


def _whiten_item(x, mean, std, lo, hi):
    if lo is not None:
        x = np.minimum(np.maximum(x, lo), hi)
    d = x - mean
    scale = np.where(std < np.float32(1e-6), np.float32(1.0), std)
    out = d / scale
    const = std < np.float32(1e-3)
    return np.where(const & (np.abs(d) > np.float32(2.0) * std), np.float32(0.0), out)


def _relu(x):
    return np.maximum(x, np.float32(0.0))


def _cell_arrays(cell: Mapping[str, Any]) -> Dict[str, Any]:
    f32 = lambda v: np.asarray(v, dtype=np.float32)  # noqa: E731
    sd = {k: f32(v) for k, v in cell["state_dict"].items()}
    nm = {k: f32(v) for k, v in cell["feature_norms"].items()}
    bd = {k: f32(v) for k, v in (cell.get("feature_bounds") or {}).items()}
    return {
        "sd": sd,
        "nm": nm,
        "bd": bd,
        "t": np.float32(max(abs(float(cell["temperature"])), 0.1)),
    }


def query_embedding(c: Mapping[str, Any], q: np.ndarray) -> np.ndarray:
    sd, nm, bd = c["sd"], c["nm"], c["bd"]
    x = _whiten(q, nm["q_mean"], nm["q_std"], bd.get("q_lo"), bd.get("q_hi"))
    h = _relu(sd["q_proj.0.weight"] @ x + sd["q_proj.0.bias"])
    h = _relu(sd["q_proj.2.weight"] @ h + sd["q_proj.2.bias"])
    return sd["q_proj.4.weight"] @ h + sd["q_proj.4.bias"]


def item_embeddings(c: Mapping[str, Any], items: np.ndarray) -> np.ndarray:
    """(keys, embed) of raw item features (keys, item dim)."""
    sd, nm, bd = c["sd"], c["nm"], c["bd"]
    x = _whiten_item(items, nm["i_mean"], nm["i_std"], bd.get("i_lo"), bd.get("i_hi"))
    h = _relu(x @ sd["i_proj.0.weight"].T + sd["i_proj.0.bias"])
    return h @ sd["i_proj.2.weight"].T + sd["i_proj.2.bias"]


def _score_sets(u: Universe) -> List[Tuple[np.ndarray, List[Dict[str, Any]]]]:
    """Candidate sets that rank every config once as the first member of
    its key, split by non-temporal hints so that each set's availability
    is no larger than that of any shipped set holding its configs."""
    out = []
    for r in range(u.members.shape[1]):
        col = u.members[:, r]
        col = col[col >= 0]
        for a in (False, True):
            for b in (False, True):
                sel = col[(u.nt_a[col] == a) & (u.nt_b[col] == b)]
                if sel.size:
                    out.append((sel, [u.kernels[i] for i in sel]))
    return out


def _init_worker(data: bytes, hardware: DeviceHardware, u: Universe) -> None:
    tw = ev.tilewright_module()
    model = ev.load_model_bytes(data, tw)
    parsed = mlrec.read_model(data)
    info = tw.describe(model)
    by_label = {c["label"]: c for c in parsed["cells"]}
    cells = [
        _cell_arrays(by_label[str(tw.cell_label(model, i))])
        for i in range(int(info.n_cells))
    ]
    hw = ev.make_hardware(tw, hardware)
    any_problem = ev.make_problem(
        tw,
        dict(
            m=256, n=256, k=256, batch=1, a_transpose="N", b_transpose="N",
            a_dtype="half", b_dtype="half", c_dtype="half", d_dtype="half",
            mi_dtype="half",
        ),
    )  # fmt: skip
    first = u.members[u.view_first, 0]
    reps = [ev.make_config(tw, u.kernels[i], int(i)) for i in first]
    items = np.array(
        [list(tw.compute_features(model, any_problem, c, hw).item) for c in reps],
        dtype=np.float32,
    ).reshape(len(reps), -1)
    _W.clear()
    _W.update(
        tw=tw,
        model=model,
        hw=hw,
        cells=cells,
        item_emb=[item_embeddings(c, items) for c in cells],
        sets=[
            (idx, ev.candidate_set(tw, model, kernels, hardware))
            for idx, kernels in _score_sets(u)
        ],
        rep=reps[0],
        n=u.n,
        contexts={},
    )


def _score_chunk(jobs: Sequence[ScoreJob]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _W
    tw, model, hw = w["tw"], w["model"], w["hw"]
    n_views = w["item_emb"][0].shape[0] if w["item_emb"] else 0
    emb = np.full((len(jobs), n_views), np.nan, dtype=np.float32)
    full = np.full((len(jobs), w["n"]), np.nan, dtype=np.float32)
    cell = np.full(len(jobs), -1, dtype=np.int32)
    for j, job in enumerate(jobs):
        # The engine ranks a context with the features of its scoring context
        # (tilewright.scoring_context: another CU budget for one outside the
        # model's CU range only by its budget) and ranks nothing for a
        # schedule the model does not list.
        entry = w["contexts"].get(job.context)
        if entry is None:
            ctx = ev.make_context(tw, job.context)
            at = tw.scoring_context(model, ctx, hw)
            entry = w["contexts"][job.context] = (ctx, at)
        ctx, at = entry
        p = ev.make_problem(tw, job.problem)
        ci = int(tw.route(model, p))
        if ci < 0 or at is None:
            continue
        cell[j] = ci
        c = w["cells"][ci]
        q = np.asarray(
            tw.compute_features(model, p, w["rep"], hw, at).query, dtype=np.float32
        )
        with np.errstate(all="ignore"):
            e = (w["item_emb"][ci] @ query_embedding(c, q)) / c["t"]
        emb[j] = np.where(np.isfinite(e), e, -np.inf)
        for idx, cs in w["sets"]:
            for r in cs.rank(p, hw, len(idx), ctx):
                if not r.scored:
                    break
                full[j, idx[int(r.config_index)]] = r.score
    return emb, full, cell


def engine_scores(
    mv: ModelView, u: Universe, jobs: Sequence[ScoreJob], workers: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(embedding term per GEMM x view, -inf when non-finite, NaN when the
    model does not serve it; full score per GEMM x config as its key's first
    member, NaN when unscored; serving cell per GEMM, -1 when none)."""
    chunks = [jobs[i : i + 64] for i in range(0, len(jobs), 64)]
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(chunks)),
            initializer=_init_worker,
            initargs=(mv.data, mv.hardware, u),
        ) as pool:
            parts = list(pool.map(_score_chunk, chunks))
    else:
        _init_worker(mv.data, mv.hardware, u)
        parts = [_score_chunk(c) for c in chunks]
    if not parts:
        return (
            np.zeros((0, u.n_views), np.float32),
            np.zeros((0, u.n), np.float32),
            np.zeros(0, np.int32),
        )
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


# ── feasibility ──────────────────────────────────────────────────────────────


def feasible_keys(
    u: Universe,
    problems: Sequence[Mapping[str, Any]],
    *,
    lds_bytes: int,
    rules: Mapping[str, bool],
    nt_a: bool,
    nt_b: bool,
) -> np.ndarray:
    """(GEMMs, keys) mask of `lib.feasibility.passes_gates` on every key's
    kernel shape, vectorized; `problems` are `lib.features` problem kwargs."""
    out = np.zeros((len(problems), u.n_keys), dtype=bool)
    groups: Dict[Tuple[Any, ...], List[int]] = {}
    for i, p in enumerate(problems):
        t = tuple(
            str(p[f]) for f in ("a_dtype", "b_dtype", "a_transpose", "b_transpose")
        )
        groups.setdefault(t, []).append(i)
    s = u.shape
    for (a_dt, b_dt, ta, tb), rows in groups.items():
        rows_a = np.array(rows)
        la, lb = feasibility.lds_bits(a_dt), feasibility.lds_bits(b_dt)
        if la <= 0 or lb <= 0:
            continue
        lds_ok = (
            s["mt_m"] * s["mt_k"] * (la / 8.0) + s["mt_n"] * s["mt_k"] * (lb / 8.0)
        ) <= float(lds_bytes)
        a_t, b_t = feasibility._is_t(ta), feasibility._is_t(tb)
        ra, rb = feasibility.rule_bits(a_dt), feasibility.rule_bits(b_dt)
        for lo in range(0, len(rows_a), _CHUNK):
            r = rows_a[lo : lo + _CHUNK]
            M, N, K, B = (
                np.array([int(problems[i][f]) for i in r], dtype=np.int64)[:, None]
                for f in ("m", "n", "k", "batch")
            )
            ok = np.broadcast_to(lds_ok, (len(r), u.n_keys)).copy()
            if rules.get("small_batched_rule", True):
                sb = (M <= 256) & (N <= 256) & (K < 1024) & (B != 1)
                ok &= ~(sb & ((s["mt_m"] < M) | (s["mt_n"] < N)))
            if rules.get("dot2_rule", True):
                d2 = (s["mi_m"] == 1) & (s["mi_n"] == 1) & (s["mi_k"] == 64)
                ok &= ~(d2 & (M > 2))
            if rules.get("leading_free_rule", True) and (
                fs.normalize_dtype_name(a_dt) != "int8x4"
            ):
                if not a_t:
                    ok &= ~(M < s["grvw_a"])
                if b_t:
                    ok &= ~(N < s["grvw_b"])
            if rules.get("non_temporal_rule", True):
                nt = feasibility.NON_TEMPORAL
                cha, chb = s["cache_hints_a"], s["cache_hints_b"]
                hinted = (cha != 0) | (chb != 0)
                aligned = ((K * ra) % 1024 == 0) & ((s["mt_k"] * ra) % 1024 == 0)
                to_b = (
                    (M <= s["mt_m"] * 2)
                    & (not b_t)
                    & ((N * rb) // np.maximum(M * ra, 1) > 5)
                )
                to_a = (
                    ~to_b
                    & (N <= s["mt_n"] * 2)
                    & a_t
                    & ((M * ra) // np.maximum(N * rb, 1) > 5)
                )
                fail_aligned = (
                    (to_b & nt_b & (chb != nt))
                    | (to_a & nt_a & (cha != nt))
                    | (~to_b & ~to_a & hinted)
                )
                ok &= ~np.where(aligned, fail_aligned, hinted)
            out[r] = ok
    return out


# ── measurements ─────────────────────────────────────────────────────────────


@dataclass
class Table:
    """Score and cost matrices of one GEMM set over a universe. `cost` is
    what a pick of each config is charged (`lib.evaluate.pick_cost`; inf:
    the runtime does not run it), `fallback_us` the runtime's fallback pick
    (NaN without one) and `charge` what an unserved GEMM costs the greedy
    objective (the fallback, else the slowest measurement)."""

    gemms: List[Any]
    contexts: List[str]
    problems: List[Dict[str, Any]]
    winner: np.ndarray
    cost: np.ndarray
    fallback_us: np.ndarray
    charge: np.ndarray
    emb: np.ndarray
    full: np.ndarray
    cell: np.ndarray
    lds_bytes: int
    rules: Dict[str, bool]
    cascade: int
    has_wl: np.ndarray
    wl: np.ndarray
    u: Universe
    _feas: Dict[Tuple[bool, bool], np.ndarray] = field(default_factory=dict)
    _ord: Optional[np.ndarray] = None
    _inv: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.gemms)

    @property
    def served(self) -> np.ndarray:
        return self.cell >= 0

    def feas(self, flags: Tuple[bool, bool]) -> np.ndarray:
        if not self.rules.get("non_temporal_rule", True):
            flags = (False, False)
        got = self._feas.get(flags)
        if got is None:
            got = self._feas[flags] = feasible_keys(
                self.u,
                self.problems,
                lds_bytes=self.lds_bytes,
                rules=self.rules,
                nt_a=flags[0],
                nt_b=flags[1],
            )
        return got

    def order(self) -> Tuple[np.ndarray, np.ndarray]:
        """(views of every GEMM by embedding term, best first; each view's
        place in that order)."""
        if self._ord is None:
            e = np.where(np.isnan(self.emb), -np.inf, self.emb)
            dt = np.int16 if self.u.n_views < np.iinfo(np.int16).max else np.int32
            o = np.argsort(-e, axis=1, kind="stable").astype(dt)
            inv = np.empty_like(o)
            np.put_along_axis(
                inv, o.astype(np.int64), np.arange(o.shape[1], dtype=dt)[None, :], 1
            )
            self._ord, self._inv = o, inv
        return self._ord, self._inv

    def wl_rows(self, rows: np.ndarray) -> np.ndarray:
        return self.wl[np.maximum(self.cell[rows], 0)] & self.served[rows][:, None]


def cost_rows(
    g: Any, u_pos: Mapping[int, int], n: int, sids: np.ndarray, pick_measurement: str
) -> Tuple[np.ndarray, float]:
    """(charged cost per universe config, fallback latency or NaN) of one
    `run_analysis.Gemm`, as the stages charge picks: a kernel off the GEMM's
    planned list at the slowest latency the GEMM measured."""
    screened_miss = pick_measurement not in ev.WARMUP_COUNTS
    worst = float(g.us.max())
    row_cost = np.where(g.skip & screened_miss, worst, g.us)
    pos = np.array([u_pos.get(int(s), -1) for s in g.sids], dtype=np.int64)
    inside = pos >= 0
    cost = np.full(n, np.inf)
    cost[pos[inside]] = row_cost[inside]
    if pick_measurement != "any" and g.listed is not None:
        off = ~np.isin(sids, np.fromiter(g.listed, dtype=np.int64))
        off[pos[inside]] = False
        cost[off] = worst
    fb = math.nan
    if g.fallback is not None:
        hit = np.flatnonzero(g.sids == g.fallback)
        if hit.size:
            fb = float(g.us[hit[0]])
    return cost, fb


def build_table(
    gemms: Sequence[Any],
    u: Universe,
    mv: ModelView,
    *,
    pick_measurement: str,
    workers: int = 1,
    context_label: Any = None,
) -> Table:
    """Table of `run_analysis.Gemm`s (their engine scores are computed
    here)."""
    if pick_measurement not in ev.PICK_MEASUREMENTS:
        raise ValueError(f"pick_measurement {pick_measurement!r}")
    from .run_analysis import context_label as default_label

    label = context_label or default_label
    u_pos = {int(s): i for i, s in enumerate(u.sids)}
    n = len(gemms)
    cost = np.full((n, u.n), np.inf, dtype=np.float32)
    fb = np.full(n, np.nan)
    charge = np.empty(n)
    winner = np.empty(n)
    problems = []
    jobs = []
    for i, g in enumerate(gemms):
        c, f = cost_rows(g, u_pos, u.n, u.sids, pick_measurement)
        cost[i] = c
        fb[i] = f
        charge[i] = f if math.isfinite(f) else float(g.us.max())
        winner[i] = float(g.winner_us)
        prob = fs.problem_kwargs_from_row(g.row)
        problems.append(prob)
        jobs.append(ScoreJob(prob, g.context))
    emb, full, cell = engine_scores(mv, u, jobs, workers)
    has_wl, wl = whitelist_masks(mv, u)
    return Table(
        gemms=list(gemms),
        contexts=[label(g.context) for g in gemms],
        problems=problems,
        winner=winner,
        cost=cost,
        fallback_us=fb,
        charge=charge,
        emb=emb,
        full=full,
        cell=cell,
        lds_bytes=int(mv.hardware.lds_bytes),
        rules=dict(mv.rules),
        cascade=int(mv.cascade),
        has_wl=has_wl,
        wl=wl,
        u=u,
    )


# ── simulated picks ──────────────────────────────────────────────────────────


@dataclass
class _Best:
    key: np.ndarray
    score: np.ndarray
    pos: np.ndarray
    cost: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "_Best":
        return cls(
            np.full(n, -1, np.int64),
            np.full(n, -np.inf, np.float32),
            np.full(n, _NO_POS, np.int64),
            np.full(n, np.inf),
        )

    @property
    def valid(self) -> np.ndarray:
        return self.key >= 0

    def at(self, idx: np.ndarray) -> "_Best":
        return _Best(self.key[idx], self.score[idx], self.pos[idx], self.cost[idx])

    def put(self, rows: np.ndarray, other: "_Best") -> None:
        self.key[rows] = other.key
        self.score[rows] = other.score
        self.pos[rows] = other.pos
        self.cost[rows] = other.cost


def _where(m: np.ndarray, a: _Best, b: _Best) -> _Best:
    return _Best(
        np.where(m, a.key, b.key),
        np.where(m, a.score, b.score),
        np.where(m, a.pos, b.pos),
        np.where(m, a.cost, b.cost),
    )


def _beats(score, pos, other: _Best) -> np.ndarray:
    """A runnable key of (score, first runnable config) is ranked before
    `other` (keys of equal score interleave their members in config
    order)."""
    return (
        ~other.valid
        | (score > other.score)
        | ((score == other.score) & (pos < other.pos))
    )


class State:
    """Every GEMM's pick state under one shipped set (and non-temporal
    availability `flags`, normally that of the set): the kept keys, the
    last view the cascade keeps, the best two runnable kept keys and the
    best one outside the view of the best."""

    def __init__(
        self,
        t: Table,
        present: np.ndarray,
        flags: Optional[Tuple[bool, bool]] = None,
        *,
        picks: bool = True,
    ) -> None:
        self.t = t
        self.present = present.copy()
        self.flags = flags if flags is not None else t.u.flags(present)
        self._present_keys()
        if not picks:
            return
        n, k = t.n, t.u.n_keys
        self.kept = np.zeros((n, k), dtype=bool)
        self.capped = np.zeros(n, dtype=bool)
        self.rank_d = np.full(n, t.u.n_views, dtype=np.int64)
        self.dview = np.full(n, -1, dtype=np.int64)
        self.wlmode = np.zeros(n, dtype=bool)
        self.p1 = _Best.empty(n)
        self.p2 = _Best.empty(n)
        self.p3 = _Best.empty(n)
        self.refresh(np.arange(n))

    def _present_keys(self) -> None:
        """First present config per key, and per view a present key (-1:
        none)."""
        u = self.t.u
        self.fp = u.first_present(self.present)
        self.vkey = np.full(u.n_views, -1, dtype=np.int64)
        has = np.flatnonzero(self.fp >= 0)
        self.vkey[u.view[has[::-1]]] = has[::-1]

    def _view_of(self, key: np.ndarray) -> np.ndarray:
        return np.where(key >= 0, self.t.u.view[np.maximum(key, 0)], -1)

    @property
    def charged(self) -> np.ndarray:
        return np.where(self.p1.valid, self.p1.cost, self.t.charge)

    @property
    def deployed(self) -> np.ndarray:
        """Cost of every GEMM's deployed pick (NaN: neither a model pick
        nor a fallback)."""
        return np.where(self.p1.valid, self.p1.cost, self.t.fallback_us)

    def objective(self) -> float:
        ok = self.t.winner > 0
        return float(np.log(self.t.winner[ok] / self.charged[ok]).sum())

    def key_summary(self, g: np.ndarray, k: np.ndarray):
        """(full score, first runnable present config, its cost) of keys
        `k` on GEMMs `g`."""
        t = self.t
        fpk = self.fp[k]
        score = np.where(fpk >= 0, t.full[g, np.maximum(fpk, 0)], np.nan)
        m = t.u.members[k]
        valid = m >= 0
        mm = np.where(valid, m, 0)
        c = t.cost[g[:, None], mm]
        ok = valid & self.present[mm] & np.isfinite(c)
        first = ok.argmax(axis=1)
        has = ok.any(axis=1)
        ar = np.arange(len(k))
        pos = np.where(has, mm[ar, first], _NO_POS)
        rcost = np.where(has, c[ar, first], np.inf)
        return score, pos, rcost

    def tier(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(tier mask, rows in whitelist mode) of rows `r`."""
        t = self.t
        f = t.feas(self.flags)[r] & (self.fp >= 0)[None, :] & t.served[r][:, None]
        wlm = np.zeros(len(r), dtype=bool)
        if t.has_wl.any():
            wt = f & t.wl_rows(r)
            wlm = t.has_wl[np.maximum(t.cell[r], 0)] & t.served[r] & wt.any(axis=1)
            f = np.where(wlm[:, None], wt, f)
        return f, wlm

    def refresh(self, rows: np.ndarray) -> None:
        t = self.t
        w = t.cascade
        for lo in range(0, len(rows), _CHUNK):
            r = rows[lo : lo + _CHUNK]
            b = len(r)
            f, wlm = self.tier(r)
            capped = np.zeros(b, dtype=bool)
            rank_d = np.full(b, t.u.n_views, dtype=np.int64)
            dview = np.full(b, -1, dtype=np.int64)
            if w > 0:
                fv = t.u.any_by_view(f)
                o = t.order()[0][r].astype(np.int64)
                s = np.take_along_axis(fv, o, axis=1)
                cum = np.cumsum(s, axis=1, dtype=np.int32)
                keep = np.zeros_like(fv)
                np.put_along_axis(keep, o, s & (cum <= w), axis=1)
                kept = np.where(wlm[:, None], f, f & keep[:, t.u.view])
                capped = ~wlm & (cum[:, -1] >= w)
                j = (s & (cum == w)).argmax(axis=1)
                rank_d = np.where(capped, j, t.u.n_views)
                dview = np.where(capped, o[np.arange(b), j], -1)
            else:
                kept = f
            self.kept[r] = kept
            self.capped[r], self.rank_d[r], self.dview[r] = capped, rank_d, dview
            self.wlmode[r] = wlm
            gi, ki = np.nonzero(kept)
            p1, p2, p3 = _Best.empty(b), _Best.empty(b), _Best.empty(b)
            if gi.size:
                score, pos, rcost = self.key_summary(r[gi], ki)
                ok = np.isfinite(score) & (pos != _NO_POS)
                gi, ki, score, pos, rcost = (
                    gi[ok],
                    ki[ok],
                    score[ok],
                    pos[ok],
                    rcost[ok],
                )
                o = np.lexsort((pos, -score, gi))
                gi, ki, score, pos, rcost = gi[o], ki[o], score[o], pos[o], rcost[o]
                starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
                if starts.size:
                    g0 = gi[starts]
                    p1.put(
                        g0,
                        _Best(ki[starts], score[starts], pos[starts], rcost[starts]),
                    )
                    nxt = starts + 1
                    two = nxt < len(gi)
                    two[two] = gi[nxt[two]] == g0[two]
                    s2 = nxt[two]
                    p2.put(gi[s2], _Best(ki[s2], score[s2], pos[s2], rcost[s2]))
                    vk = t.u.view[ki]
                    group = np.cumsum(np.r_[True, gi[1:] != gi[:-1]]) - 1
                    other = np.flatnonzero(vk != vk[starts][group])
                    _, first = np.unique(group[other], return_index=True)
                    s3 = other[first]
                    p3.put(gi[s3], _Best(ki[s3], score[s3], pos[s3], rcost[s3]))
            self.p1.put(r, p1)
            self.p2.put(r, p2)
            self.p3.put(r, p3)

    def relevant(self, r: np.ndarray, cands: np.ndarray) -> np.ndarray:
        """(rows, candidates) mask of the absent configs whose addition can
        change the rows' kept keys or picks."""
        t = self.t
        kc = t.u.key[cands]
        vc = t.u.view[kc]
        newk = self.fp[kc] < 0
        newv = newk & (self.vkey[vc] < 0)
        rel = np.zeros((len(r), len(cands)), dtype=bool)
        if newv.any():
            cols = np.flatnonzero(newv)
            kn = kc[cols]
            f = t.feas(self.flags)[np.ix_(r, kn)] & t.served[r][:, None]
            if t.cascade > 0:
                inv = t.order()[1][np.ix_(r, vc[cols])]
                enter = f & (~self.capped[r][:, None] | (inv < self.rank_d[r][:, None]))
            else:
                enter = f.copy()
            if t.has_wl.any():
                cell = np.maximum(t.cell[r], 0)
                wl = t.wl[np.ix_(cell, kn)] & t.served[r][:, None]
                wlm = self.wlmode[r][:, None]
                has = t.has_wl[cell][:, None]
                enter = np.where(wlm, f & wl, enter | (f & wl & has))
            rel[:, cols] = enter
        joins = newk & ~newv
        if joins.any():
            cols = np.flatnonzero(joins)
            rel[:, cols] = self.kept[np.ix_(r, self.vkey[vc[cols]])]
        if (~newk).any():
            cols = np.flatnonzero(~newk)
            rel[:, cols] = self.kept[np.ix_(r, kc[cols])]
        return rel

    def new_costs(self, g: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Charged cost of GEMMs `g` once config `c` (relevant to each) is
        added."""
        t = self.t
        k = t.u.key[c]
        sc = t.full[g, c]
        cc = t.cost[g, c].astype(np.float64)
        run_c = np.isfinite(sc) & np.isfinite(cc)
        fb = t.charge[g]
        p1, p2, p3 = self.p1.at(g), self.p2.at(g), self.p3.at(g)
        out = np.empty(len(g))
        isnew = self.fp[k] < 0
        newv = isnew & (self.vkey[t.u.view[k]] < 0)
        joins = isnew & ~newv
        if joins.any():
            # Another launch class of a kept view: one more kept key.
            i = np.flatnonzero(joins)
            wins = run_c[i] & _beats(sc[i], c[i], p1.at(i))
            out[i] = np.where(wins, cc[i], np.where(p1.valid[i], p1.cost[i], fb[i]))
        if newv.any():
            i = np.flatnonzero(newv)
            gi, ki = g[i], k[i]
            switch = np.zeros(len(i), dtype=bool)
            if t.has_wl.any():
                switch = (
                    t.has_wl[np.maximum(t.cell[gi], 0)]
                    & ~self.wlmode[gi]
                    & t.wl[np.maximum(t.cell[gi], 0), ki]
                )
            drop = self.capped[gi] & (self.dview[gi] == self._view_of(p1.key[i]))
            base = _where(drop, p3.at(i), p1.at(i))
            wins = run_c[i] & _beats(sc[i], c[i], base)
            normal = np.where(wins, cc[i], np.where(base.valid, base.cost, fb[i]))
            out[i] = np.where(switch, np.where(run_c[i], cc[i], fb[i]), normal)
        if (~isnew).any():
            i = np.flatnonzero(~isnew)
            gi, ki, ci = g[i], k[i], c[i]
            ks0, rpos0, rcost0 = self.key_summary(gi, ki)
            ks = np.where(ci < self.fp[ki], sc[i], ks0)
            better = np.isfinite(cc[i]) & (ci < rpos0)
            rpos = np.where(better, ci, rpos0)
            rcost = np.where(better, cc[i], rcost0)
            ok = np.isfinite(ks) & (rpos != _NO_POS)
            other = _where(p1.key[i] == ki, p2.at(i), p1.at(i))
            wins = ok & _beats(ks, rpos, other)
            out[i] = np.where(wins, rcost, np.where(other.valid, other.cost, fb[i]))
        return out

    def deltas(
        self, rows: np.ndarray, cands: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, candidate index, gain in log selection efficiency) of every
        relevant pair; pairs not listed gain nothing."""
        out_r, out_c, out_d = [], [], []
        charged = self.charged
        for lo in range(0, len(rows), _CHUNK):
            r = rows[lo : lo + _CHUNK]
            pi, pc = np.nonzero(self.relevant(r, cands))
            if not pi.size:
                continue
            g = r[pi]
            new = self.new_costs(g, cands[pc])
            ok = self.t.winner[g] > 0
            d = np.where(ok, np.log(charged[g]) - np.log(new), 0.0)
            out_r.append(g)
            out_c.append(pc)
            out_d.append(d)
        if not out_r:
            e = np.zeros(0, np.int64)
            return e, e, np.zeros(0)
        return np.concatenate(out_r), np.concatenate(out_c), np.concatenate(out_d)

    def add(self, c: int) -> np.ndarray:
        """Ship config `c` (same availability flags); returns the rows whose
        state changed."""
        one = np.array([c])
        rows = np.concatenate(
            [
                r[self.relevant(r, one)[:, 0]]
                for r in np.array_split(
                    np.arange(self.t.n), max(1, -(-self.t.n // (8 * _CHUNK)))
                )
            ]
        )
        self.present[c] = True
        self._present_keys()
        self.refresh(rows)
        return rows


def simulate(t: Table, present: np.ndarray) -> State:
    return State(t, present)


def picks(state: State) -> np.ndarray:
    """Universe position of every GEMM's model pick (-1: none)."""
    return np.where(state.p1.valid, state.p1.pos, -1)


def ceiling(t: Table, present: np.ndarray) -> np.ndarray:
    """Per GEMM, the cost under perfect scoring of the shipped set as the
    runtime serves it: the cheapest tier key (no cascade), each key served
    by its first runnable member under the pick measurement, else the
    fallback (NaN without one). `identity_ceiling` is the model-free
    bound of the same set."""
    st = State(t, present, picks=False)
    out = t.fallback_us.copy()
    for lo in range(0, t.n, _CHUNK):
        r = np.arange(lo, min(lo + _CHUNK, t.n))
        f, _ = st.tier(r)
        gi, ki = np.nonzero(f)
        if not gi.size:
            continue
        _, pos, rcost = st.key_summary(r[gi], ki)
        best = np.full(len(r), np.inf)
        np.minimum.at(best, gi, rcost)
        hit = np.isfinite(best)
        out[r[hit]] = best[hit]
    return out


# ── selection ────────────────────────────────────────────────────────────────


def identity_ceiling(t: Table, present: np.ndarray) -> Dict[str, Any]:
    """`lib.identity_ceiling` of the shipped set on `t`'s GEMMs, grouped as
    the universe groups it (the engine's grouping for the model): every
    measured kernel counts as admitted, and neither the feasibility tier,
    the whitelists nor the pick measurement apply. Geomean, losing GEMMs
    and groups overall, and the geomean per execution context."""
    from . import identity_ceiling as ic

    u = t.u
    idx = np.flatnonzero(present)
    key_of = {int(u.sids[i]): int(u.key[i]) for i in idx}
    dicts = [gemm_dict(g) for g in t.gemms]
    shipped = u.shipped(present)
    pools = {ev.pool_key(d): shipped for d in dicts}
    c = ic.ceiling(dicts, lambda k: key_of[int(k["sol_idx_global"])], "engine", pools)
    out = c.to_dict()
    per: Dict[str, List[float]] = {}
    for d, label in zip(dicts, t.contexts):
        v = c.per_gemm.get(ev.gemm_key(d))
        if v is not None:
            per.setdefault(label, []).append(v)
    out["per_context"] = {label: _geo(np.array(v)) for label, v in sorted(per.items())}
    return out


def win_order(t: Table) -> np.ndarray:
    """Bank positions by the selection GEMMs each wins (its charged cost is
    the GEMM's fastest), then by summed closeness to the fastest, then by
    shipped order."""
    u = t.u
    bank = u.bank
    c = t.cost[:, bank].astype(np.float64)
    w = t.winner[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        close = np.where(np.isfinite(c) & (w > 0), w / c, 0.0)
    wins = (np.isfinite(c) & (c <= w * (1 + 1e-9))).sum(axis=0)
    return bank[np.lexsort((bank, -close.sum(axis=0), -wins))]


@dataclass
class Greedy:
    order: List[int]
    objective: List[float]


def greedy_order(
    t: Table,
    max_size: int,
    *,
    tie_order: Optional[np.ndarray] = None,
    progress: Any = None,
) -> Greedy:
    """Forward selection of up to `max_size` bank configs, each step adding
    the one with the largest gain in summed log selection efficiency over
    `t`'s GEMMs (ties to the earlier one of `tie_order`, default
    `win_order`)."""
    u = t.u
    cands = tie_order if tie_order is not None else win_order(t)
    cands = np.asarray(cands, dtype=np.int64)
    max_size = min(int(max_size), len(cands))
    present = u.present()
    avail = np.ones(len(cands), dtype=bool)

    def flags_of(c: np.ndarray, base: Tuple[bool, bool]) -> List[Tuple[bool, bool]]:
        if not t.rules.get("non_temporal_rule", True):
            return [base] * len(c)
        return [(base[0] or bool(u.nt_a[i]), base[1] or bool(u.nt_b[i])) for i in c]

    class Group:
        """Candidates sharing the availability flags they ship with: their
        state and (GEMM, candidate) gains."""

        def __init__(self, f: Tuple[bool, bool], cols: np.ndarray) -> None:
            self.cols = cols
            self.st = State(t, present, f)
            self.d = np.zeros((t.n, len(cols)), dtype=np.float32)
            self.fill(np.arange(t.n))
            self.colsum = self.d.sum(axis=0, dtype=np.float64)

        def fill(self, rows: np.ndarray) -> None:
            live = np.flatnonzero(avail[self.cols])
            self.d[rows] = 0.0
            rr, cc, dd = self.st.deltas(rows, cands[self.cols[live]])
            self.d[rr, live[cc]] = dd

        def add(self, c: int) -> None:
            rows = self.st.add(c)
            if rows.size:
                self.colsum -= self.d[rows].sum(axis=0, dtype=np.float64)
                self.fill(rows)
                self.colsum += self.d[rows].sum(axis=0, dtype=np.float64)

    def build() -> Tuple[Tuple[bool, bool], Dict[Tuple[bool, bool], Group]]:
        base = u.flags(present)
        fl = flags_of(cands, base)
        groups = {}
        for f in dict.fromkeys(fl):
            groups[f] = Group(f, np.flatnonzero([x == f for x in fl]))
        if base not in groups:
            groups[base] = Group(base, np.zeros(0, dtype=np.int64))
        return base, groups

    base, groups = build()
    order: List[int] = []
    objective: List[float] = [groups[base].st.objective()]
    while len(order) < max_size:
        main = groups[base].st.objective()
        gain = np.full(len(cands), -np.inf)
        for f, gr in groups.items():
            offset = gr.st.objective() - main if f != base else 0.0
            gain[gr.cols] = offset + gr.colsum
        gain[~avail] = -np.inf
        j = int(np.argmax(gain))
        c = int(cands[j])
        avail[j] = False
        order.append(c)
        present[c] = True
        if u.flags(present) != base:
            base, groups = build()
        else:
            for gr in groups.values():
                gr.add(c)
        objective.append(groups[base].st.objective())
        if progress is not None:
            progress(len(order), objective[-1])
    return Greedy(order=order, objective=objective)


# ── metrics ──────────────────────────────────────────────────────────────────


def _geo(x: np.ndarray) -> Optional[float]:
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if x.size else None


def set_metrics(
    t: Table, present: np.ndarray, *, worst: int = 0, with_ceiling: bool = True
) -> Dict[str, Any]:
    """Selection efficiency of the deployed picks of `t` under a shipped
    set (geomean over the GEMMs with a model pick or a fallback, as
    `lib.evaluate` counts them), overall and per execution context, with
    the GEMMs losing more than 1 %, the perfect-scoring ceiling and the
    identity ceiling (`identity_ceiling`)."""
    if t.n == 0:
        return {"n": 0, "sel_eff": None}
    st = State(t, present)
    cost = st.deployed
    ok = np.isfinite(cost) & (cost > 0) & (t.winner > 0)
    eff = np.full(t.n, np.nan)
    eff[ok] = t.winner[ok] / cost[ok]
    ceil = ceiling(t, present) if with_ceiling else None

    def block(mask: np.ndarray) -> Dict[str, Any]:
        e = eff[mask & ok]
        out = {
            "n": int(e.size),
            "sel_eff": _geo(e),
            "n_loss_1pct": int((e < LOSS).sum()),
            "n_fallback": int((mask & ok & ~st.p1.valid).sum()),
        }
        if ceil is not None:
            okc = mask & np.isfinite(ceil) & (ceil > 0) & (t.winner > 0)
            out["ceiling"] = _geo(t.winner[okc] / ceil[okc])
        return out

    out = block(np.ones(t.n, dtype=bool))
    if with_ceiling:
        ident = identity_ceiling(t, present)
        out["identity_ceiling"] = ident["geomean"]
        out["identity"] = ident
    out["n_unservable"] = int((~ok).sum())
    ctx = np.array(t.contexts, dtype=object)
    out["per_context"] = {c: block(ctx == c) for c in sorted(set(t.contexts))}
    if worst:
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(eff[idx], kind="stable")][:worst]
        pk = picks(st)
        out["worst"] = [
            {
                "m": int(t.problems[i]["m"]),
                "n": int(t.problems[i]["n"]),
                "k": int(t.problems[i]["k"]),
                "batch": int(t.problems[i]["batch"]),
                "context": t.contexts[i],
                "sel_eff": float(eff[i]),
                "pick_sol_idx": int(t.u.sids[pk[i]]) if pk[i] >= 0 else None,
                "served_by": "model" if pk[i] >= 0 else "origami_fallback",
            }
            for i in idx
            if eff[i] < LOSS
        ]
    return out


def recommend(
    rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[float] = BUDGETS_US,
    tolerance: float = RECOMMEND_TOLERANCE,
) -> Dict[str, Any]:
    """Per selection-time budget: the smallest subset size whose held-out
    selection efficiency is within `tolerance` of the best among the sizes
    whose p99 ranking time fits the budget. `rows` carry `n_bank`,
    `held_out.sel_eff` and `latency_us.p99`."""
    out: Dict[str, Any] = {}
    for budget in budgets:
        fit = [
            r
            for r in rows
            if (r.get("latency_us") or {}).get("p99") is not None
            and r["latency_us"]["p99"] <= budget
            and (r.get("held_out") or {}).get("sel_eff") is not None
        ]
        key = f"{budget:g}"
        if not fit:
            out[key] = None
            continue
        best = max(r["held_out"]["sel_eff"] for r in fit)
        pick = min(
            (r for r in fit if r["held_out"]["sel_eff"] >= best - tolerance),
            key=lambda r: r["n_bank"],
        )
        out[key] = {
            "size": pick["size"],
            "n_bank": pick["n_bank"],
            "held_out_sel_eff": pick["held_out"]["sel_eff"],
            "best_held_out_sel_eff": best,
            "p99_us": pick["latency_us"]["p99"],
        }
    return out


# ── evaluate_model parity ────────────────────────────────────────────────────


def gemm_dict(g: Any) -> Dict[str, Any]:
    """A `lib.evaluate` GEMM dict of a `run_analysis.Gemm`."""
    d = dict(g.row)
    d["winner_us"] = float(g.winner_us)
    d["candidates"] = [
        {
            "sol_idx_global": int(s),
            "us": float(us),
            "is_skip": bool(sk),
            "rank": -1,
            "is_origami_pick": (
                1 if g.fallback is not None and int(s) == g.fallback else 0
            ),
        }
        for s, us, sk in zip(g.sids, g.us, g.skip)
        if int(s) >= 0
    ]
    if g.listed is not None:
        d["listed"] = g.listed
    return d


def parity(
    t: Table,
    present: np.ndarray,
    rows: Sequence[int],
    *,
    pick_measurement: str,
    data: bytes,
    hardware: DeviceHardware,
    tw: Any = None,
) -> Dict[str, Any]:
    """Simulated picks against `lib.evaluate.evaluate_model` on GEMMs
    `rows` of `t` with the shipped set `present`."""
    rows = list(rows)
    if not rows:
        return {"n": 0, "n_same": 0, "mismatches": []}
    st = State(t, present)
    shipped = t.u.shipped(present)
    dicts = [gemm_dict(t.gemms[i]) for i in rows]
    pools = {ev.pool_key(d): shipped for d in dicts}
    res = ev.evaluate_model(
        data,
        dicts,
        hardware=hardware,
        pools=pools,
        tw=tw,
        pick_measurement=pick_measurement,
    ).results
    sim = picks(st)
    same, bad = 0, []
    for i, r in zip(rows, res):
        mine = int(t.u.sids[sim[i]]) if sim[i] >= 0 else None
        theirs = r.pick_sol_idx if r.served_by == "model" else None
        cost_mine = st.deployed[i]
        cost_theirs = r.pick_us if r.pick_us is not None else math.nan
        if mine == theirs and (
            (math.isnan(cost_mine) and math.isnan(cost_theirs))
            or abs(cost_mine - cost_theirs) <= 1e-6 * max(1.0, abs(cost_theirs))
        ):
            same += 1
        else:
            bad.append({"row": int(i), "simulated": mine, "engine": theirs})
    return {"n": len(rows), "n_same": same, "mismatches": bad[:20]}


def cascade_parity(
    t: Table,
    present: np.ndarray,
    rows: Sequence[int],
    *,
    data: bytes,
    hardware: DeviceHardware,
    tw: Any = None,
) -> Dict[str, Any]:
    """The keys the engine fully scores (`CandidateSet.rank`, one config
    scored) against the simulated cascade, on GEMMs `rows`."""
    tw = tw or ev.tilewright_module()
    model = ev.load_model_bytes(data, tw)
    idx = np.flatnonzero(present)
    cs = ev.candidate_set(tw, model, [t.u.kernels[i] for i in idx], hardware)
    hw = ev.make_hardware(tw, hardware)
    st = State(t, present)
    same, bad = 0, []
    for i in rows:
        g = t.gemms[i]
        ranked = cs.rank(
            ev.make_problem(tw, t.problems[i]), hw, 1, ev.make_context(tw, g.context)
        )
        got = {int(t.u.key[idx[int(r.config_index)]]) for r in ranked if r.scored}
        kept = np.flatnonzero(st.kept[i])
        want = {int(k) for k in kept[np.isfinite(t.full[i, st.fp[kept]])]}
        if got == want:
            same += 1
        else:
            bad.append(
                {
                    "row": int(i),
                    "only_engine": len(got - want),
                    "only_sim": len(want - got),
                }
            )
    return {"n": len(list(rows)), "n_same": same, "mismatches": bad[:20]}


# ── selection time ───────────────────────────────────────────────────────────


def bench_inputs(
    problem: Mapping[str, Any],
    hardware: DeviceHardware,
    shapes: Sequence[Tuple[int, int, int, int]],
    kernels: Sequence[Mapping[str, Any]],
) -> str:
    """The inputs file of scripts/experiments/bench_shipped_rank.cpp (and
    bench_selection_time.cpp) for
    `lib.features` problem kwargs, the device and the shipped kernels."""
    dtypes = [
        DATATYPE_VALUE[fs.normalize_dtype_name(problem[f])]
        for f in ("a_dtype", "b_dtype", "c_dtype", "d_dtype", "mi_dtype")
    ]
    trans = [
        0 if fs._transpose_bit(problem[f]) else 1
        for f in ("a_transpose", "b_transpose")
    ]
    lines = [
        "PROBLEM " + " ".join(str(v) for v in dtypes + trans),
        f"HARDWARE {hardware.n_cu} {hardware.lds_bytes} {hardware.l2_bytes}",
        f"SHAPES {len(shapes)}",
    ]
    lines += [f"{m} {n} {k} {b}" for m, n, k, b in shapes]
    lines.append(f"CONFIGS {len(kernels)}")
    lines += [
        " ".join(
            [str(int(c[f])) for f in BENCH_FIELDS]
            + [f"{k}={int(v)}" for k, v in (c.get("attributes") or {}).items()]
        )
        for c in kernels
    ]
    return "\n".join(lines) + "\n"


def bench_shapes(
    t: Table, limit: int = 1000, seed: int = 0
) -> List[Tuple[int, int, int, int]]:
    """Up to `limit` distinct problem shapes of `t`, a seeded sample."""
    shapes = sorted(
        {(int(p["m"]), int(p["n"]), int(p["k"]), int(p["batch"])) for p in t.problems}
    )
    if len(shapes) > limit:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(shapes), size=limit, replace=False))
        shapes = [shapes[i] for i in keep]
    return shapes


BENCH_TARGET = "bench_shipped_rank"


def build_bench(
    source_dir: Path, build_dir: Path, jobs: int = 8, target: str = BENCH_TARGET
) -> Path:
    """Build `target` (Release) from `source_dir` (scripts/experiments);
    returns the binary. The engine is built without its re-rank memo
    (TILEWRIGHT_LEVER_MEMO), which can answer every other one of the
    bench's repeated rankings of a shape, so that every timed call ranks
    afresh; the prefetch stays as the engine ships it."""
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTILEWRIGHT_LEVER_MEMO=OFF",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            target,
            "-j",
            str(jobs),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return build_dir / target


_ROW = re.compile(
    r"^candidate_set\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
)


def parse_bench(text: str) -> Optional[Dict[str, float]]:
    """The candidate_set row of bench_shipped_rank's (or
    bench_selection_time's) output."""
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if m:
            v = [float(x) for x in m.groups()[1:]]
            return {
                "shapes": int(m.group(1)),
                "min": v[0],
                "p50": v[1],
                "mean": v[2],
                "p90": v[3],
                "p99": v[4],
                "max": v[5],
            }
    return None


def run_bench(
    binary: Path,
    model_file: Path,
    inputs_file: Path,
    *,
    core: int,
    iters: int = 200,
    warmup: int = 50,
) -> Dict[str, float]:
    """p50 / p99 (and more) microseconds of one cached `CandidateSet::rank`
    call, single-threaded on CPU `core`."""
    proc = subprocess.run(
        [
            str(binary),
            "--model", str(model_file),
            "--inputs", str(inputs_file),
            "--iters", str(int(iters)),
            "--warmup", str(int(warmup)),
            "--min-scored", "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        preexec_fn=lambda: os.sched_setaffinity(0, {int(core)}),
    )  # fmt: skip
    got = parse_bench(proc.stdout)
    if got is None:
        raise RuntimeError(f"no candidate_set row in the output of {binary}")
    return got


_REDUCTIONS = {"none": 0, "tree": 1, "parallel": 2, "atomic": 3}


def launch_parity(
    binary: Path,
    inputs_file: Path,
    problem: Mapping[str, Any],
    shapes: Sequence[Tuple[int, int, int, int]],
    kernels: Sequence[Mapping[str, Any]],
    hardware: DeviceHardware,
    launch_id: str,
    n_shapes: int = 50,
) -> Dict[str, Any]:
    """bench_shipped_rank's launch plans and launch keys of `kernels` (the
    configs of `inputs_file`) on the first `n_shapes` shapes against
    `lib.hipblaslt_launch` for the model's launch id, exclusive use."""
    from . import hipblaslt_launch as hl

    parts = str(launch_id).split(":")
    if len(parts) != 3 or parts[0] != "hipblaslt" or parts[2] != str(hl.RULE_VERSION):
        return {"launch_model": launch_id, "n": 0, "n_same": 0, "checked": False}
    arch = parts[1]
    out = subprocess.run(
        [
            str(binary), "--inputs", str(inputs_file), "--dump-plans", str(int(n_shapes)),
            "--dump-arch", arch,
        ],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    ctx = fs.ExecutionContext()
    n = same = 0
    first: Optional[Dict[str, Any]] = None
    for line in out.splitlines():
        _, si, ci, dyn, grid, red, key = line.split()
        m, nn, k, b = shapes[int(si)]
        c = kernels[int(ci)]
        want = hl.launch_plan(
            dict(m=m, n=nn, k=k, batch=b, mi_dtype=problem.get("mi_dtype")),
            c, c.get("attributes"), ctx, hardware.n_cu, arch, hardware.num_xcd,
        )  # fmt: skip
        ok = (bool(int(dyn)), int(grid), int(red), int(key)) == (
            want.dynamic_queue,
            want.grid_wgs,
            _REDUCTIONS[want.partial_reduction],
            hl.launch_key(c, c.get("attributes")),
        )
        n += 1
        same += ok
        if not ok and first is None:
            first = {"shape": [m, nn, k, b], "sol_idx": int(c["sol_idx_global"])}
    return {
        "launch_model": launch_id,
        "n": n,
        "n_same": same,
        "checked": True,
        "first_mismatch": first,
    }


# ── report ───────────────────────────────────────────────────────────────────


def _pct(v: Any) -> str:
    return "&ndash;" if v is None else f"{100 * float(v):.2f}%"


def _us(v: Any) -> str:
    return "&ndash;" if v is None else f"{float(v):.1f}"


def report_html(doc: Mapping[str, Any], esc: Any = None) -> str:
    """The report's "Shipped candidate set" section of a shipped_subset.json
    document."""
    import html

    esc = esc or (lambda s: html.escape(str(s)))
    rows = doc.get("sizes") or []
    held = doc.get("held_out") or {}
    sel = doc.get("selection") or {}
    has_hc = any(r.get("holdout_contexts") for r in rows)
    head = (
        "<tr><th>bank kernels</th><th>configs</th><th>selection</th><th>evaluation</th>"
        "<th>evaluation &gt;1% loss</th><th>ceiling (evaluation)</th>"
        "<th>identity ceiling (evaluation)</th>"
        + ("<th>held-out contexts</th>" if has_hc else "")
        + "<th>p50 &micro;s</th><th>p99 &micro;s</th>"
        "<th>win-count order (evaluation)</th></tr>"
    )
    body = []
    for r in rows:
        h = r.get("held_out") or {}
        s = r.get("selection") or {}
        lat = r.get("latency_us") or {}
        wb = (r.get("wins_baseline") or {}).get("held_out") or {}
        hc = r.get("holdout_contexts") or {}
        body.append(
            f'<tr><td>{esc(r["size"])}</td><td>{r["n_configs"]:,}</td>'
            f'<td>{_pct(s.get("sel_eff"))}</td><td>{_pct(h.get("sel_eff"))}</td>'
            f'<td>{h.get("n_loss_1pct", 0):,} / {h.get("n", 0):,}</td>'
            f'<td>{_pct(h.get("ceiling"))}</td>'
            f'<td>{_pct(h.get("identity_ceiling"))}</td>'
            + (f'<td>{_pct(hc.get("sel_eff"))}</td>' if has_hc else "")
            + f'<td>{_us(lat.get("p50"))}</td>'
            f'<td>{_us(lat.get("p99"))}</td><td>{_pct(wb.get("sel_eff"))}</td></tr>'
        )
    rec = doc.get("recommendation") or {}
    rec_txt = []
    for budget, v in rec.items():
        if v is None:
            rec_txt.append(f"<li>{esc(budget)} &micro;s: no measured size fits</li>")
        else:
            rec_txt.append(
                f"<li>{esc(budget)} &micro;s: <b>{esc(v['size'])}</b> bank kernels "
                f"({_pct(v['held_out_sel_eff'])} evaluation, best that fits "
                f"{_pct(v['best_held_out_sel_eff'])}, p99 {_us(v['p99_us'])} &micro;s)</li>"
            )
    per_size = []
    labels: List[str] = []
    for r in rows:
        per = dict((r.get("held_out") or {}).get("per_context") or {})
        for c, v in (
            (r.get("holdout_contexts") or {}).get("per_context") or {}
        ).items():
            per[f"{c} (held-out context)"] = v
        labels += [c for c in per if c not in labels]
        per_size.append((r, per))
    ctx = [
        f"<tr><td>{esc(r['size'])}</td>"
        + "".join(
            (
                f"<td>{_pct(per[c].get('sel_eff'))} ({per[c].get('n', 0):,})</td>"
                if c in per
                else "<td>&ndash;</td>"
            )
            for c in labels
        )
        + "</tr>"
        for r, per in per_size
        if per
    ]
    ctx_head = (
        "<tr><th>bank kernels</th>"
        + "".join(f"<th>{esc(c)} (GEMMs)</th>" for c in labels)
        + "</tr>"
    )
    notes = "".join(f"<li>{esc(n)}</li>" for n in doc.get("notes") or [])
    par = doc.get("parity") or {}
    par_txt = ", ".join(
        f"{esc(k)} {v.get('n_same', 0)}/{v.get('n', 0)}"
        for k, v in par.items()
        if isinstance(v, Mapping)
    )
    return (
        f'<p class="cap">The pool ({doc.get("pool", {}).get("n", 0):,} kernels) plus '
        f"the first B bank kernels ({doc.get('bank', {}).get('n', 0):,} in the bank) of a "
        f"greedy forward selection on {sel.get('n_gemms', 0):,} training-side GEMMs, "
        f"evaluated on {held.get('n_gemms', 0):,} GEMMs it did not choose on: "
        f"{esc(held.get('label') or held.get('source', ''))}. Held-out execution "
        "contexts, which no model trained on, are a column of their own. "
        "Picks are the shipped bundle's simulated "
        f"engine picks charged by <code>{esc(doc.get('pick_measurement', ''))}</code>; "
        "the ceiling is perfect scoring of the same set as the runtime serves it, the "
        "identity ceiling perfect scoring of its candidates (lib/identity_ceiling.py: "
        "every measured kernel admitted, no feasibility tier or whitelist); ranking "
        "time is one cached "
        "CandidateSet::rank call at the bundle's cascade width (bench_shipped_rank, "
        "one pinned core, warm).</p>"
        f'<div class="scroll"><table><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        + (f"<ul>{notes}</ul>" if notes else "")
        + f"<p><b>Recommended</b> (smallest B within {RECOMMEND_TOLERANCE:g} of the best "
        f"evaluation selection efficiency under the budget):</p><ul>{''.join(rec_txt)}</ul>"
        + (
            "<details><summary>evaluation selection efficiency per execution context"
            f'</summary><div class="scroll"><table><thead>{ctx_head}</thead>'
            f'<tbody>{"".join(ctx)}</tbody></table></div></details>'
            if ctx
            else ""
        )
        + (
            f'<p class="mut">simulated vs engine picks: {par_txt}</p>'
            if par_txt
            else ""
        )
    )
