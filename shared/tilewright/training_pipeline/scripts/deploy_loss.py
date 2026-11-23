#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where a stage05 bundle loses selection efficiency between training and
deployment.

stage05 reports two validation numbers: the student's (`val_sel_eff`: the
weights rounded as --weight-dtype stores them, scored in PyTorch over each
validation GEMM's measured candidates) and the deployed one
(`deployed_val`: the engine ranking the library pool under the model's
serving options, lib/evaluate.py). This tool recomputes both on stage05's
own validation GEMMs and evaluates deploy variants (weight dtype x cascade
width) so that the gap can be split per GEMM.

  prepare   load stage05's inputs exactly as stage05 did (its loader with
            its repeat aggregation, blocklist and the library pool's
            kernels alone, pool order and validation split, from
            metrics.json `args` and the config snapshot they name) and
            cache the validation GEMMs with the library pool
  student   the student's picks per validation GEMM, as stage05 measures
            them, for each weight dtype
  grid      deployed picks per validation GEMM for each weight dtype and
            cascade width (0 = every feasible key fully scored)
  report    the variant table and the per-GEMM decomposition of the gap

Usage:
  deploy_loss.py prepare --stage05-dir DIR --work-dir DIR
      [--hardware n_cu,lds_bytes,l2_bytes] [--pci-chip-id ID]
  deploy_loss.py student --work-dir DIR [--dtypes fp32,bf16,int8,int4]
  deploy_loss.py grid --work-dir DIR [--dtypes ...] [--cascades 16,32,64,0]
  deploy_loss.py report --work-dir DIR [--deployed int4:32]

--hardware and --pci-chip-id replace the KFD probe when the stage runs
without a visible device; by default the hardware stage05 recorded in
metrics.json is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_THIS_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lib import correctness as cx  # noqa: E402
from lib import evaluate as ev  # noqa: E402
from lib import hardware as hwlib  # noqa: E402
from lib import repeats  # noqa: E402

DTYPES = ("fp32", "bf16", "int8", "int4")
CASCADES = (16, 32, 64, 0)
PREPARED = "prepared.pkl"


def _stage05() -> Any:
    from stages import stage05_train

    return stage05_train


def _geomean(values: Sequence[float]) -> float:
    return ev.geomean(values)


def _wlog_mean(pairs: Sequence[Tuple[Optional[float], int]]) -> float:
    """stage05's `wlog_mean`: weighted geometric mean of finite positives."""
    return _stage05().wlog_mean(list(pairs))


# ── prepare ──────────────────────────────────────────────────────────────────


def run_config(
    stage05_dir: Path,
    hardware: Optional[Sequence[int]],
    pci_chip_id: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(metrics.json, args stage05 ran with, pipeline config with the
    hardware block set to what stage05 recorded or to `hardware`)."""
    metrics = json.loads((stage05_dir / "metrics.json").read_text())
    args = metrics["args"]
    cfg: Dict[str, Any] = {}
    if args.get("config_yaml"):
        cfg = yaml.safe_load(Path(args["config_yaml"]).read_text()) or {}
    block = dict(cfg.get("hardware") or {})
    rec = metrics.get("hardware") or {}
    if hardware is not None:
        rec = dict(zip(hwlib.HARDWARE_KEYS, hardware))
    for k in hwlib.HARDWARE_KEYS:
        if k in rec:
            block[k] = int(rec[k])
    if pci_chip_id is not None:
        block["pci_chip_id"] = int(pci_chip_id)
    cfg["hardware"] = block
    return metrics, args, cfg


def prepare(ns: argparse.Namespace) -> None:
    s5 = _stage05()
    sc = s5.sc
    stage05_dir = Path(ns.stage05_dir)
    hw_override = [int(v) for v in ns.hardware.split(",")] if ns.hardware else None
    metrics, args, cfg = run_config(stage05_dir, hw_override, ns.pci_chip_id)
    hardware = hwlib.device_hardware(cfg, int(args.get("hardware_device") or 0))
    split_tree = sc.load_split_tree(
        args.get("prior_round_dir") or [], args.get("current_round_dir")
    )
    identity = (
        ev.library_identity(
            args["library_dir"],
            args["library_stem"],
            cfg,
            int(args.get("hardware_device") or 0),
        )
        if args.get("library_stem")
        else None
    )
    library = (
        ev.library_pool(
            args["library_dir"],
            args["library_stem"],
            cfg,
            int(args.get("hardware_device") or 0),
        )
        if args.get("library_stem")
        else None
    )
    cells, load_stats = s5.load_enriched_chunks(
        args["enriched_csv_dir"],
        False,
        split_tree,
        identity,
        aggregation=args.get("repeat_aggregation") or repeats.PREDATING_AGGREGATION,
        blocklist=cx.config_blocklist(cfg),
        pool_indices=(
            None if library is None else [int(k["sol_idx_global"]) for k in library]
        ),
    )
    if metrics.get("catalog") != "v3":
        cells, _ = s5.drop_non_exclusive(cells, hardware.n_cu)
    all_gemms = [g for gs in cells.values() for g in gs]
    pools = ev.pools_for(all_gemms, library)
    ev.order_candidates_by_pool(all_gemms, pools)
    frac, seed = float(args["validation_frac"]), int(args["seed"])
    trained = set((metrics.get("per_cell_val_sel_eff") or {}).keys())
    val: Dict[str, List[dict]] = {}
    n_train: Dict[str, int] = {}
    for label, gemms in cells.items():
        if label not in trained:
            continue
        train_g, val_g = s5.split_validation(gemms, frac, seed)
        val[label] = val_g
        n_train[label] = len(train_g)
    out = {
        "stage05_dir": str(stage05_dir),
        "args": args,
        "cfg": cfg,
        "hardware": hardware,
        "pools": pools,
        "identity": identity,
        "val": val,
        "n_train": n_train,
        "load_stats": load_stats,
    }
    work = Path(ns.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    with (work / PREPARED).open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_val = sum(len(v) for v in val.values())
    want = (metrics.get("validation") or {}).get("n_gemms")
    print(
        f"prepared {len(val)} cells, {n_val} validation GEMMs "
        f"(stage05: {want}), pool sizes "
        f"{ {'/'.join(k): len(v) for k, v in pools.items()} }"
    )
    if want is not None and int(want) != n_val:
        raise SystemExit(f"validation GEMMs differ from stage05's ({want})")


def load_prepared(work: Path) -> Dict[str, Any]:
    with (work / PREPARED).open("rb") as f:
        return pickle.load(f)


def load_bundle(stage05_dir: Path) -> Dict[str, Any]:
    import torch

    return torch.load(stage05_dir / "models.pt", map_location="cpu", weights_only=False)


# ── student ──────────────────────────────────────────────────────────────────


def student_picks(ns: argparse.Namespace) -> None:
    """Per validation GEMM, the candidate the student picks (stage05's
    `measure`: stored weights, every measured candidate, first wins ties)."""
    import torch

    s5 = _stage05()
    work = Path(ns.work_dir)
    prep = load_prepared(work)
    bundle = load_bundle(Path(prep["stage05_dir"]))
    names = (bundle["q_names"], bundle["i_names"], bundle["x_names"])
    args = prep["args"]
    constants = hwlib.arch_constants(args["arch"])
    catalog = s5.training_catalog(argparse.Namespace(**args), prep["cfg"])
    torch.set_num_threads(int(ns.threads))
    out: Dict[str, Dict[str, List[dict]]] = {d: {} for d in ns.dtypes}
    for label, val_g in sorted(prep["val"].items()):
        entry = bundle["models"][label]
        arrays = s5.build_cell_tensors(
            val_g, *names, prep["hardware"], constants, catalog
        )
        vq_raw, vi_raw, vx_raw, gidx, us, win, _ = arrays
        vq, vi, vx = s5.apply_whiten(
            vq_raw, vi_raw, vx_raw, entry["feature_norms"], entry.get("feature_bounds")
        )
        pairs = s5._PairSet((vq, vi, vx, gidx, us, win), "cpu")
        student = s5.GenericTwoTower(
            q_dim=len(names[0]),
            i_dim=len(names[1]),
            x_dim=len(names[2]),
            embed_dim=int(entry["embed_dim"]),
            hidden_dim=int(entry["hidden_dim"]),
            inter_hidden=int(entry["inter_hidden"]),
        )
        student.load_exported(entry["state_dict"])
        student.eval()
        starts = np.concatenate([[0], np.cumsum(pairs.counts)[:-1]])
        for dtype in ns.dtypes:
            student.fake_quant = None if dtype == "fp32" else dtype
            with torch.no_grad():
                scores = pairs.scores(student).cpu().numpy().astype(np.float64)
            rows = []
            for gi, g in enumerate(val_g):
                s, n = int(starts[gi]), int(pairs.counts[gi])
                block = scores[s : s + n]
                j = int(np.argmax(block)) if n else -1
                c = g["candidates"][j] if j >= 0 else None
                rows.append(
                    {
                        "pick_sol_idx": None if c is None else int(c["sol_idx_global"]),
                        "pick_us": None if c is None else float(c["us"]),
                        "winner_us": float(g["winner_us"]),
                    }
                )
            out[dtype][label] = rows
        print(f"student {label}: {len(val_g)} GEMMs", flush=True)
    with (work / "student.pkl").open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    metrics = json.loads((Path(prep["stage05_dir"]) / "metrics.json").read_text())
    for dtype in ns.dtypes:
        print(f"student {dtype}: {summarize_student(out[dtype], prep, metrics)}")


def summarize_student(
    rows_by_cell: Mapping[str, List[dict]],
    prep: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    per_cell = {}
    for label, rows in rows_by_cell.items():
        per_cell[label] = _geomean(
            r["winner_us"] / r["pick_us"] for r in rows if r["pick_us"]
        )
    out = {
        "cells_weighted": _wlog_mean(
            [(per_cell[c], len(prep["val"][c])) for c in per_cell]
        ),
        "gemm_geomean": _geomean(
            r["winner_us"] / r["pick_us"]
            for rows in rows_by_cell.values()
            for r in rows
            if r["pick_us"]
        ),
    }
    if metrics is not None:
        ref = metrics.get("per_cell_val_sel_eff") or {}
        diffs = [
            abs(per_cell[c] - float(ref[c])) for c in per_cell if ref.get(c) is not None
        ]
        out["max_cell_diff_vs_stage05"] = max(diffs) if diffs else float("nan")
    return out


# ── grid ─────────────────────────────────────────────────────────────────────


def variant_bundle(bundle: Mapping[str, Any], cascade: int) -> Dict[str, Any]:
    b = dict(bundle)
    b["serving"] = dict(bundle.get("serving") or {})
    b["serving"]["cascade_keys"] = int(cascade)
    return b


def grid(ns: argparse.Namespace) -> None:
    from lib import mlrec

    work = Path(ns.work_dir)
    prep = load_prepared(work)
    bundle = load_bundle(Path(prep["stage05_dir"]))
    args = prep["args"]
    constants = hwlib.arch_constants(args["arch"])
    labels = sorted(prep["val"])
    gemms = [g for c in labels for g in prep["val"][c]]
    owner = [c for c in labels for _ in prep["val"][c]]
    suffix = ""
    if ns.plan:
        gemms = [dict(g) for g in gemms]
        n = ev.attach_plan(gemms, ev.read_plan(ns.plan), prep.get("identity"))
        print(f"plan: {n} of {len(gemms)} GEMMs benched from a kernel list")
        suffix = "_plan"
    tw = ev.tilewright_module()
    for dtype in ns.dtypes:
        for cascade in ns.cascades:
            path = work / f"grid_{dtype}_c{cascade}{suffix}.pkl"
            if path.exists() and not ns.force:
                continue
            data = mlrec.write_model(
                variant_bundle(bundle, cascade), None, args["arch"], constants, dtype
            )
            out = ev.evaluate_model(
                data,
                gemms,
                hardware=prep["hardware"],
                pools=prep["pools"],
                labels=list(bundle["models"].keys()),
                tw=tw,
                pick_measurement=args.get("pick_measurement") or "warmup",
            )
            rows = [
                {
                    "cell": owner[i],
                    "pick_sol_idx": r.pick_sol_idx,
                    "pick_us": r.pick_us,
                    "winner_us": r.winner_us,
                    "pick_rank": r.pick_rank,
                    "n_scored": r.n_scored,
                    "served_by": r.served_by,
                    "miss": r.miss,
                    "model_cell": r.model_cell,
                }
                for i, r in enumerate(out.results)
            ]
            with path.open("wb") as f:
                pickle.dump(
                    {"rows": rows, "n_routing_mismatch": out.n_routing_mismatch,
                     "model_bytes": len(data)},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )  # fmt: skip
            s = summarize_rows(rows, prep)
            print(
                f"{dtype} c{cascade}: cells_weighted={s['cells_weighted']:.4f} "
                f"gemm_geomean={s['gemm_geomean']:.4f} evaluated={s['n_evaluated']} "
                f"model_bytes={len(data)} routing_mismatch={out.n_routing_mismatch}",
                flush=True,
            )


def top1(ns: argparse.Namespace) -> None:
    """The engine's first pick (rank_top k=1, what the runtime runs when its
    predicates pass) per validation GEMM for each variant, whether or not the
    GEMM measured it."""
    from lib import features as fs
    from lib import mlrec

    work = Path(ns.work_dir)
    prep = load_prepared(work)
    bundle = load_bundle(Path(prep["stage05_dir"]))
    args = prep["args"]
    constants = hwlib.arch_constants(args["arch"])
    labels = sorted(prep["val"])
    gemms = [g for c in labels for g in prep["val"][c]]
    tw = ev.tilewright_module()
    hw = ev.make_hardware(tw, prep["hardware"])
    for dtype in ns.dtypes:
        for cascade in ns.cascades:
            data = mlrec.write_model(
                variant_bundle(bundle, cascade), None, args["arch"], constants, dtype
            )
            model = ev.load_model_bytes(data, tw)
            poolset = ev.PoolSet(tw, model, prep["pools"], prep["hardware"])
            out = []
            for g in gemms:
                key = ev.pool_key(g)
                cs = poolset.candidate_set(key)
                pool = poolset.pools[key]
                prob = ev.make_problem(tw, fs.problem_kwargs_from_row(g))
                top = cs.rank_top(prob, hw, 1, ev.make_context(tw, ev.context_of(g)))
                out.append(
                    int(pool[int(top[0].config_index)]["sol_idx_global"])
                    if top
                    else None
                )
            with (work / f"top1_{dtype}_c{cascade}.pkl").open("wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"top1 {dtype} c{cascade}: {sum(v is not None for v in out)}",
                flush=True,
            )


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], prep: Mapping[str, Any]
) -> Dict[str, Any]:
    """stage05's two aggregates of deployed rows: per-cell geomeans over the
    evaluated GEMMs weighted by every validation GEMM of the cell
    (`global_deployed_val_sel_eff`), and the geomean over evaluated GEMMs
    (`validation.sel_eff`)."""
    by_cell: Dict[str, List[float]] = defaultdict(list)
    ses = []
    for r in rows:
        if r["pick_us"] and r["winner_us"] > 0:
            se = r["winner_us"] / r["pick_us"]
            by_cell[r["cell"]].append(se)
            ses.append(se)
    return {
        "cells_weighted": _wlog_mean(
            [(_geomean(by_cell[c]), len(prep["val"][c])) for c in prep["val"]]
        ),
        "gemm_geomean": _geomean(ses),
        "n_evaluated": len(ses),
        "per_cell": {c: _geomean(v) for c, v in by_cell.items()},
    }


# ── report ───────────────────────────────────────────────────────────────────


def _variant(text: str) -> Tuple[str, int]:
    dtype, cascade = text.split(":")
    return dtype, int(cascade)


def _load_grid(work: Path, suffix: str = "") -> Dict[Tuple[str, int], List[dict]]:
    out = {}
    for path in sorted(work.glob(f"grid_*_c*{suffix}.pkl")):
        stem = path.stem[len("grid_") :]
        if suffix:
            if not stem.endswith(suffix):
                continue
            stem = stem[: -len(suffix)]
        elif stem.endswith("_plan"):
            continue
        dtype, c = stem.rsplit("_c", 1)
        with path.open("rb") as f:
            out[(dtype, int(c))] = pickle.load(f)["rows"]
    return out


def _sel(r: Mapping[str, Any]) -> Optional[float]:
    if r.get("pick_us") and r["winner_us"] > 0:
        return r["winner_us"] / r["pick_us"]
    return None


def _gm(values: Sequence[Optional[float]]) -> float:
    return _geomean([v for v in values if v])


def _pts(log_sum: float, n: int) -> float:
    """Selection-efficiency points that a sum of per-GEMM log latency ratios
    (pick / reference) costs a geomean over `n` GEMMs."""
    return 100.0 * (1.0 - math.exp(-log_sum / n)) if n else float("nan")


def report(ns: argparse.Namespace) -> None:
    work = Path(ns.work_dir)
    prep = load_prepared(work)
    labels = sorted(prep["val"])
    gemms = [g for c in labels for g in prep["val"][c]]
    ctx = [int(ev.context_of(g).cu_budget) for g in gemms]
    grid_rows = _load_grid(work)
    plan_rows = _load_grid(work, "_plan")
    with (work / "student.pkl").open("rb") as f:
        student = pickle.load(f)
    stu = {d: [r for c in labels for r in student[d][c]] for d in student}
    n = len(gemms)
    lines: List[str] = []
    emit = lines.append

    # Variant grid.
    common = [i for i in range(n) if all(_sel(rows[i]) for rows in grid_rows.values())]
    contexts = sorted(set(ctx))
    emit(f"# {n} validation GEMMs; common set (evaluable in every variant): "
         f"{len(common)}")  # fmt: skip
    head = ["variant", "stage05 cell-wtd", "gemm geomean", "evaluated",
            "not eval.", "pick off top-1", "common set"]  # fmt: skip
    head += [f"common cu{c}" for c in contexts]
    if plan_rows:
        head += ["plan: geomean all", "plan: misses"]
    emit("| " + " | ".join(head) + " |")
    emit("|" + "---|" * len(head))
    for d in ("student",):
        for dtype in DTYPES:
            if dtype not in stu:
                continue
            rows = stu[dtype]
            cells = defaultdict(list)
            for c, r in zip((c for c in labels for _ in prep["val"][c]), rows):
                cells[c].append(_sel(r))
            cw = _wlog_mean([(_gm(v), len(prep["val"][c])) for c, v in cells.items()])
            vals = [f"student {dtype}", f"{cw:.4f}", f"{_gm([_sel(r) for r in rows]):.4f}",
                    str(sum(1 for r in rows if _sel(r))), "0", "-",
                    f"{_gm([_sel(rows[i]) for i in common]):.4f}"]  # fmt: skip
            vals += [
                f"{_gm([_sel(rows[i]) for i in common if ctx[i] == c]):.4f}"
                for c in contexts
            ]
            if plan_rows:
                vals += ["-", "-"]
            emit("| " + " | ".join(vals) + " |")
    for dtype in DTYPES:
        for cascade in CASCADES:
            rows = grid_rows.get((dtype, cascade))
            if rows is None:
                continue
            s = summarize_rows(rows, prep)
            off = sum(1 for r in rows if (r.get("pick_rank") or 0) > 0)
            vals = [f"{dtype} c{cascade or 'full'}", f"{s['cells_weighted']:.4f}",
                    f"{s['gemm_geomean']:.4f}", str(s["n_evaluated"]),
                    str(n - s["n_evaluated"]), str(off),
                    f"{_gm([_sel(rows[i]) for i in common]):.4f}"]  # fmt: skip
            vals += [
                f"{_gm([_sel(rows[i]) for i in common if ctx[i] == c]):.4f}"
                for c in contexts
            ]
            if plan_rows:
                pr = plan_rows.get((dtype, cascade))
                if pr is None:
                    vals += ["-", "-"]
                else:
                    vals += [
                        f"{_gm([_sel(r) for r in pr]):.4f}",
                        str(sum(1 for r in pr if r.get("miss"))),
                    ]
            emit("| " + " | ".join(vals) + " |")

    # Weakest cells under the deployed variant.
    dep = _variant(ns.deployed)
    ref = _variant(ns.reference)
    if dep in grid_rows:
        emit("")
        emit(f"# {ns.weakest} weakest cells under {ns.deployed} (stage05's per-cell "
             f"deployed metric), common-set geomeans per variant")  # fmt: skip
        per_cell = summarize_rows(grid_rows[dep], prep)["per_cell"]
        weakest = sorted(per_cell, key=lambda c: per_cell[c])[: ns.weakest]
        owner = [c for c in labels for _ in prep["val"][c]]
        cols = [k for k in sorted(grid_rows) if k[1] in CASCADES]
        head = ["cell", "val GEMMs", "not eval.", f"{ns.deployed} (stage05)",
                "student int4"] + [f"{d} c{c or 'full'}" for d, c in cols]  # fmt: skip
        emit("| " + " | ".join(head) + " |")
        emit("|" + "---|" * len(head))
        common_set = set(common)
        for c in weakest:
            idx = [i for i in range(n) if owner[i] == c]
            cidx = [i for i in idx if i in common_set]
            vals = [c, str(len(idx)),
                    str(sum(1 for i in idx if not _sel(grid_rows[dep][i]))),
                    f"{per_cell[c]:.4f}",
                    f"{_gm([_sel(stu['int4'][i]) for i in cidx]):.4f}"]  # fmt: skip
            vals += [f"{_gm([_sel(grid_rows[k][i]) for i in cidx]):.4f}" for k in cols]
            emit("| " + " | ".join(vals) + " |")

    # Decomposition against the reference (fp32, full scoring).
    for w in CASCADES:
        if not w:
            continue
        fc = (ref[0], w)
        qf = (dep[0], 0)
        dw = (dep[0], w)
        need = [ref, fc, qf, dw]
        if not all(k in grid_rows for k in need):
            continue
        R, C, Q, D = (grid_rows[k] for k in need)
        emit("")
        emit(f"# decomposition at cascade {w}: reference {ref[0]} full scoring, "
             f"deployed {dep[0]} c{w}")  # fmt: skip
        both_eval = [i for i in range(n) if _sel(R[i])]
        lost = [i for i in both_eval if not _sel(D[i])]
        ev_ = [i for i in both_eval if _sel(D[i]) and _sel(C[i]) and _sel(Q[i])]
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i in ev_:
            rec = C[i]["pick_sol_idx"] == R[i]["pick_sol_idx"]
            qok = Q[i]["pick_sol_idx"] == R[i]["pick_sol_idx"]
            buckets[
                "neither" if rec and qok
                else "cascade only" if qok
                else "quantization only" if rec
                else "both"
            ].append(i)  # fmt: skip
        m = len(ev_)
        emit(f"GEMMs evaluable under the reference: {len(both_eval)}; the deployed "
             f"variant scores no measured kernel for {len(lost)} of them (the "
             f"cascade kept no measured key; recall failures of unknown cost)")  # fmt: skip
        emit(f"decomposed over the {m} GEMMs evaluable in all four variants; cost "
             f"in sel-eff points of the {m}-GEMM geomean, deployed vs reference")  # fmt: skip
        emit("| bucket | GEMMs | deployed pick != reference | cost (points) |")
        emit("|---|---|---|---|")

        def lr(rows: List[dict], i: int) -> float:
            return math.log(rows[i]["pick_us"] / R[i]["pick_us"])

        total = 0.0
        for name in ("neither", "cascade only", "quantization only", "both"):
            idx = buckets.get(name, [])
            ls = sum(lr(D, i) for i in idx)
            total += ls
            diff = sum(1 for i in idx if D[i]["pick_sol_idx"] != R[i]["pick_sol_idx"])
            emit(f"| {name} | {len(idx)} | {diff} | {_pts(ls, m):.3f} |")
        emit(f"| total | {m} | | {_pts(total, m):.3f} |")
        c_only = sum(lr(C, i) for i in ev_)
        q_only = sum(lr(Q, i) for i in ev_)
        emit(f"cascade alone ({ref[0]} c{w} vs full): {_pts(c_only, m):.3f} points; "
             f"quantization alone ({dep[0]} full vs {ref[0]} full): "
             f"{_pts(q_only, m):.3f}; interaction: "
             f"{_pts(total - c_only - q_only, m):.3f}")  # fmt: skip
        recall = sum(1 for i in both_eval if _sel(C[i])
                     and C[i]["pick_sol_idx"] == R[i]["pick_sol_idx"])  # fmt: skip
        emit(f"cascade recall of the reference pick ({ref[0]} c{w}): "
             f"{recall}/{len(both_eval)} = {recall / max(1, len(both_eval)):.4f}")  # fmt: skip

    # The engine's first picks, measured or not.
    tops: Dict[Tuple[str, int], List[Optional[int]]] = {}
    for path in sorted(work.glob("top1_*_c*.pkl")):
        dtype, c = path.stem[len("top1_") :].rsplit("_c", 1)
        with path.open("rb") as f:
            tops[(dtype, int(c))] = pickle.load(f)
    if ref in tops:
        cost: List[Dict[int, float]] = []
        for g in gemms:
            timed, screened = ev.measurements(g)
            cost.append({**screened, **timed})
        R = tops[ref]
        emit("")
        emit(f"# first picks (rank_top k=1), measured or not, against {ref[0]} full "
             f"scoring; latency ratios only where both picks are measured")  # fmt: skip
        emit(
            "| variant | top-1 measured | same top-1 as reference | "
            "both measured, different | cost on those (points of all both-measured) |"
        )
        emit("|---|---|---|---|---|")
        for k in sorted(tops, key=lambda k: (DTYPES.index(k[0]), CASCADES.index(k[1]))):
            T = tops[k]
            meas = sum(1 for i in range(n) if T[i] in cost[i])
            same = sum(1 for i in range(n) if T[i] == R[i])
            both = [i for i in range(n) if T[i] in cost[i] and R[i] in cost[i]]
            diff = [i for i in both if T[i] != R[i]]
            ls = sum(math.log(cost[i][T[i]] / cost[i][R[i]]) for i in diff)
            emit(f"| {k[0]} c{k[1] or 'full'} | {meas} | {same} | {len(diff)} | "
                 f"{_pts(ls, len(both)):.3f} (n={len(both)}) |")  # fmt: skip
        for w in CASCADES:
            fc, qf, dw = (ref[0], w), (dep[0], 0), (dep[0], w)
            if not w or not all(k in tops for k in (fc, qf, dw)):
                continue
            C, Q, D = tops[fc], tops[qf], tops[dw]
            b = defaultdict(list)
            for i in range(n):
                rec, qok = C[i] == R[i], Q[i] == R[i]
                b["neither" if rec and qok else "cascade only" if qok
                  else "quantization only" if rec else "both"].append(i)  # fmt: skip
            emit(f"cascade {w}: top-1 recall of {ref[0]} full in {ref[0]} c{w}: "
                 f"{sum(1 for i in range(n) if C[i] == R[i])}/{n}; buckets "
                 + ", ".join(f"{name} {len(b[name])} (deployed top-1 differs "
                             f"{sum(1 for i in b[name] if D[i] != R[i])})"
                             for name in ("neither", "cascade only",
                                          "quantization only", "both")))  # fmt: skip

    # Student vs engine (same weights, same dtype): the evaluation's own gap.
    if "int4" in stu and dep in grid_rows and (dep[0], 0) in grid_rows:
        emit("")
        emit("# student (PyTorch, measured candidates) vs engine, same weights")
        S, F, D = stu[dep[0]], grid_rows[(dep[0], 0)], grid_rows[dep]
        allf = [i for i in range(n) if _sel(F[i])]
        same = sum(1 for i in allf if F[i]["pick_sol_idx"] == S[i]["pick_sol_idx"])
        emit(f"{dep[0]} full scoring picks the student's kernel on {same}/{len(allf)} "
             f"GEMMs; student {_gm([_sel(S[i]) for i in allf]):.4f} vs engine "
             f"{_gm([_sel(F[i]) for i in allf]):.4f} on them")  # fmt: skip
        evd = [i for i in range(n) if _sel(D[i])]
        nev = [i for i in range(n) if not _sel(D[i])]
        emit(f"student {dep[0]} on the {len(evd)} GEMMs {ns.deployed} evaluates: "
             f"{_gm([_sel(S[i]) for i in evd]):.4f}; on the {len(nev)} it does not: "
             f"{_gm([_sel(S[i]) for i in nev]):.4f}")  # fmt: skip
    text = "\n".join(lines) + "\n"
    (work / "report.md").write_text(text)
    print(text)


# ── latency inputs ───────────────────────────────────────────────────────────


def export(ns: argparse.Namespace) -> None:
    """The bundle as model files, one per weight dtype and cascade width."""
    from lib import mlrec

    work = Path(ns.work_dir)
    prep = load_prepared(work)
    bundle = load_bundle(Path(prep["stage05_dir"]))
    args = prep["args"]
    constants = hwlib.arch_constants(args["arch"])
    for dtype in ns.dtypes:
        for cascade in ns.cascades:
            data = mlrec.write_model(
                variant_bundle(bundle, cascade), None, args["arch"], constants, dtype
            )
            path = work / f"model_{dtype}_c{cascade}.tilewright.bin"
            path.write_bytes(data)
            print(f"{path.name}: {len(data)} bytes")
    print(f"launch model id: {(bundle.get('provenance') or {}).get('launch_model')}")


def bench_inputs(ns: argparse.Namespace) -> None:
    """scripts/experiments/bench_selection_time.cpp inputs: the library pool
    and a sample of the validation GEMMs (with their cu_budget), the same
    number from every cell."""
    import random

    from lib import features as fs
    from lib.hardware import DATATYPE_VALUE

    work = Path(ns.work_dir)
    prep = load_prepared(work)
    ((key, pool),) = prep["pools"].items()
    rng = random.Random(ns.seed)
    per_cell = max(1, ns.n_shapes // max(1, len(prep["val"])))
    shapes = []
    for label in sorted(prep["val"]):
        gs = list(prep["val"][label])
        rng.shuffle(gs)
        for g in gs[:per_cell]:
            ctx = ev.context_of(g)
            shapes.append(
                (int(g["m"]), int(g["n"]), int(g["k"]), int(g["batch_count"]),
                 int(ctx.cu_budget))
            )  # fmt: skip
    g0 = next(iter(prep["val"].values()))[0]
    prob = fs.problem_kwargs_from_row(g0)
    dt = [int(DATATYPE_VALUE[fs.normalize_dtype_name(prob[f])]) for f in
          ("a_dtype", "b_dtype", "c_dtype", "d_dtype", "mi_dtype")]  # fmt: skip
    trans = [0 if prob[f] == "T" else 1 for f in ("a_transpose", "b_transpose")]
    hw = prep["hardware"]
    lines = [
        "PROBLEM " + " ".join(map(str, dt + trans)),
        f"HARDWARE {hw.n_cu} {hw.lds_bytes} {hw.l2_bytes}",
        f"SHAPES {len(shapes)}",
        *(" ".join(map(str, s)) for s in shapes),
        f"CONFIGS {len(pool)}",
    ]
    for k in pool:
        vals = [int(k[f]) for f in ev.CONFIG_FIELDS]
        attrs = [
            f"{n}={int(v)}" for n, v in sorted((k.get("attributes") or {}).items())
        ]
        lines.append(
            " ".join(map(str, vals)) + (" " + " ".join(attrs) if attrs else "")
        )
    path = work / "bench_inputs.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"{path}: {len(shapes)} shapes, {len(pool)} kernels")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--stage05-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--hardware", default=None, help="n_cu,lds_bytes,l2_bytes")
    p.add_argument("--pci-chip-id", type=lambda s: int(s, 0), default=None)
    p = sub.add_parser("student")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--dtypes", type=lambda s: s.split(","), default=list(DTYPES))
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("grid")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--dtypes", type=lambda s: s.split(","), default=list(DTYPES))
    p.add_argument(
        "--cascades",
        type=lambda s: [int(v) for v in s.split(",")],
        default=list(CASCADES),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--plan",
        default=None,
        help="stage02 plan.json: a ranked kernel off a GEMM's list is a miss "
        "(lib/evaluate.py `listed`) instead of a kernel the runtime skips",
    )
    p = sub.add_parser("top1")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--dtypes", type=lambda s: s.split(","), default=list(DTYPES))
    p.add_argument(
        "--cascades",
        type=lambda s: [int(v) for v in s.split(",")],
        default=list(CASCADES),
    )
    p = sub.add_parser("export")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--dtypes", type=lambda s: s.split(","), default=list(DTYPES))
    p.add_argument(
        "--cascades",
        type=lambda s: [int(v) for v in s.split(",")],
        default=list(CASCADES),
    )
    p = sub.add_parser("bench-inputs")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--n-shapes", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--deployed", default="int4:32", help="dtype:cascade")
    p.add_argument("--reference", default="fp32:0", help="dtype:cascade")
    p.add_argument("--weakest", type=int, default=10)
    ns = ap.parse_args(argv)
    {
        "prepare": prepare,
        "student": student_picks,
        "grid": grid,
        "top1": top1,
        "export": export,
        "bench-inputs": bench_inputs,
        "report": report,
    }[ns.cmd](ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
