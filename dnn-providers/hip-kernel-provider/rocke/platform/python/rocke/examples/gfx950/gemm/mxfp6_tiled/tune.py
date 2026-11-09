# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hill-climb with look-back over the tiled MXFP6 GEMM knobs.

Every optimization is a knob (a spec field) with a default. Each round
evaluates every single-knob move away from the current best configuration,
including moves that failed in earlier rounds: a knob that did not pay off on
its own is always re-tried on top of each newly accepted configuration. When no
single move clears the acceptance threshold, the look-back step evaluates pairs
of previously failed moves on different knobs before the climb stops.

Each evaluation runs ``bench.py`` in its own process. Optionally a second
process runs under ``rocprofv3`` to read MFMA busy cycles, the effective shader
clock and fabric read traffic. Everything is appended to ``ledger.jsonl`` in
``--ledger``, and ``report.md`` there is rewritten after every evaluation.
The ledger holds measured numbers: keep it outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Knob space. The first value of each knob is its default.
KNOBS: dict[str, list] = {
    "frag_pipeline": ["stage", "step"],
    "mfma_group": [16, 8, 4, 2, 1],
    "mfma_order": ["rows", "serp"],
    "lds_alias_scopes": [False, True],
    "a_scale_fetch": ["stage", "iter", "iter2"],
    "a_scale_path": ["direct", "lds2"],
    "as_read": ["b32", "b128x4"],
    "a_ring": ["stage", "line", "line2r"],
    "fill_split": ["none", "line1"],
    "scale_opsel": [False, True],
    "k_peel": [False, True],
    "step_vmem_order": ["reads_first", "group_late", "b_early", "refill_early", "frag_first", "frag_mid"],
    "dma_place": ["default", "even", "even_bfirst", "dense_bfirst", "pair_bfirst"],
    "sync_place": ["none", "split", "sink", "split_sink", "split_lgkm"],
    "prologue_wait": ["all", "stage0"],
    "epilogue": ["default", "sa", "pk2", "ct", "ct4", "lds", "ct_lds"],
    "store_nt": [False, True],
    "k_order": ["rot", "t10"],
    "b_release": ["full", "half"],
    "b_path": ["lds", "vgpr"],
    "dma_m0": ["default", "early", "bshare", "early_bshare"],
    "cc_passes": ["none", "dead_lanes", "m0_fill", "peep_waits", "peep_shifts", "ds_rebase", "all"],
    "cc_extra": ["none", "lds_waw+u16_zext", "acc_init_keep", "pin_vmem", "m0_cse", "all"],
    "acc_init": ["default", "early"],
    "b_layout": ["row", "kblock", "lanes"],
    "b_scale_layout": ["row", "kblock", "kblock4"],
    "b_scale_fetch": ["stage", "pair"],
    "mfma_form": ["asm_agpr", "intrinsic"],
    "stages": [2, 3],
    "lds_swizzle": ["rot16", "none"],
    "xcd_remap": ["none", "xcd_block"],
    "sched": ["none", "interleave", "iglp0", "iglp1"],
    "setprio": [False, True],
    "waves_per_eu": [1, 0],
    "agpr_alloc": ["256,256", "auto"],
    "dma_cache": [0, 1, 2],
    "acc_fence": ["none", "agpr"],
}
CUS, SIMDS_PER_CU, XCDS = 256, 4, 8
PEAK_HBM_GBPS = 8192.0
PMC = ("GRBM_GUI_ACTIVE", "SQ_VALU_MFMA_BUSY_CYCLES", "FETCH_SIZE")


def config_key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


def describe(cfg: dict, base: dict) -> str:
    diff = [f"{k}={cfg[k]}" for k in KNOBS if cfg[k] != base[k]]
    return ", ".join(diff) or "defaults"


class Tuner:
    def __init__(self, args):
        self.args = args
        self.ledger_dir = Path(args.ledger)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.ledger_dir / "ledger.jsonl"
        self.cache = Path(args.cache)
        self.records: dict[str, dict] = {}
        self.history: list[dict] = []
        self.defaults = {k: v[0] for k, v in KNOBS.items()}
        if self.ledger.exists():  # resume: never re-run a configuration
            for line in self.ledger.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("kind") == "confirm" or "config" not in rec:
                    continue
                if rec.get("shape") == self.shape and rec.get("data") == args.data:
                    self.records[config_key(rec["config"])] = rec
                    self.history.append(rec)

    @property
    def shape(self) -> list[int]:
        return [int(v) for v in self.args.shape.split(",")]

    # ------------------------------------------------------------ evaluation

    def _bench(self, cfg: dict, mode: str, extra: list[str] | None = None) -> list[str]:
        return [
            sys.executable,
            "-m",
            "rocke.examples.gfx950.gemm.mxfp6_tiled.bench",
            "--mode",
            mode,
            "--config",
            json.dumps(cfg),
            "--shape",
            self.args.shape,
            "--data",
            self.args.data,
            "--cache",
            str(self.cache),
            *(extra or []),
        ]

    def _pmc(self, cfg: dict) -> dict:
        rocprof = shutil.which("rocprofv3")
        if rocprof is None:
            return {"status": "unavailable"}
        out = Path(tempfile.mkdtemp(prefix="pmc_", dir=self.cache))
        cmd = [
            rocprof,
            "--pmc",
            *PMC,
            "--kernel-trace",
            "--output-format",
            "csv",
            "-d",
            str(out),
            "-o",
            "run",
            "--",
            *self._bench(cfg, "profile", ["--iters", "3"]),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=self.args.timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        counters, dur = {}, None
        for path in out.rglob("*counter_collection.csv"):
            rows = list(csv.DictReader(path.open(newline="")))
            rows = [r for r in rows if r.get("Kernel_Name", "").startswith("mxfp6")]
            if not rows:
                continue
            last = max(int(r.get("Dispatch_Id", 0) or 0) for r in rows)
            for r in rows:
                if int(r.get("Dispatch_Id", 0) or 0) == last:
                    counters[r["Counter_Name"]] = counters.get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
        for path in out.rglob("*kernel_trace.csv"):
            durs = [
                int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
                for r in csv.DictReader(path.open(newline=""))
                if r.get("Kernel_Name", "").startswith("mxfp6")
            ]
            if durs:
                dur = durs[-1]
        shutil.rmtree(out, ignore_errors=True)
        if not counters or not dur:
            return {"status": "no-data"}
        cycles = counters.get("GRBM_GUI_ACTIVE", 0.0) / XCDS
        res = {"status": "ok", "dur_us": dur / 1e3, "counters": counters}
        if cycles:
            res["clock_mhz"] = cycles / dur * 1e3
            res["mfma_util"] = counters.get("SQ_VALU_MFMA_BUSY_CYCLES", 0.0) / (cycles * CUS * SIMDS_PER_CU)
        if "FETCH_SIZE" in counters:
            res["hbm_gbps"] = counters["FETCH_SIZE"] * 1024 / dur
            res["hbm_util"] = res["hbm_gbps"] / PEAK_HBM_GBPS
        return res

    def evaluate(self, cfg: dict, *, move: str, parent: dict | None, round_: int, pmc: bool) -> dict:
        key = config_key(cfg)
        if key in self.records:
            rec = self.records[key]
            if pmc and "pmc" not in rec and rec.get("status") == "ok":
                rec["pmc"] = self._pmc(cfg)
                self._append(rec)
            return rec
        out = self.cache / f"eval_{len(self.records):04d}.json"
        t0 = time.time()
        try:
            proc = subprocess.run(
                self._bench(cfg, "eval", ["--out", str(out), *self._timing_args()]),
                capture_output=True,
                text=True,
                timeout=self.args.timeout,
                check=False,
            )
            rec = json.loads(out.read_text()) if out.exists() else {
                "status": "error",
                "reason": (proc.stderr or proc.stdout)[-2000:],
            }
        except subprocess.TimeoutExpired:
            rec = {"status": "timeout"}
        rec.update(
            config=cfg,
            shape=self.shape,
            data=self.args.data,
            move=move,
            parent=config_key(parent) if parent else None,
            round=round_,
            wall_s=time.time() - t0,
        )
        if pmc and rec.get("status") == "ok":
            rec["pmc"] = self._pmc(cfg)
        self.records[key] = rec
        self.history.append(rec)
        self._append(rec)
        self._log(rec)
        return rec

    def _append(self, rec: dict) -> None:
        with self.ledger.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        self.write_report()

    def _timing_args(self) -> list[str]:
        return ["--repeats", str(self.args.repeats), "--sustain-ms", str(self.args.sustain_ms)]

    def _retime(self, cfg: dict) -> float:
        """One more timing-only measurement of an already-checked configuration."""
        out = self.cache / f"retime_{time.time_ns()}.json"
        try:
            subprocess.run(
                self._bench(cfg, "eval", ["--out", str(out), "--skip-checks", *self._timing_args()]),
                capture_output=True,
                text=True,
                timeout=self.args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return 0.0
        rec = json.loads(out.read_text()) if out.exists() else {}
        return rec.get("tflops", 0.0) if rec.get("status") == "timing" else 0.0

    def _confirm(self, winner: dict, incumbent: dict) -> bool:
        """Re-time incumbent and challenger alternately; compare medians."""
        if self.args.confirm <= 0:
            return True
        w, b = [winner.get("tflops", 0.0)], [incumbent.get("tflops", 0.0)]
        for _ in range(self.args.confirm):
            b.append(self._retime(incumbent["config"]))
            w.append(self._retime(winner["config"]))

        def median(xs):
            return sorted(xs)[len(xs) // 2]

        ok = median(w) > median(b) * (1 + self.args.threshold)
        winner["tflops_confirmed"], incumbent["tflops_confirmed"] = median(w), median(b)
        with self.ledger.open("a") as fh:
            fh.write(json.dumps({
                "kind": "confirm", "winner": winner["config"], "incumbent": incumbent["config"],
                "winner_tflops": w, "incumbent_tflops": b, "accepted": ok,
            }) + "\n")
        print(f"[tune] confirm {describe(winner['config'], incumbent['config'])}: "
              f"{median(w):.1f} vs {median(b):.1f} -> {'accept' if ok else 'reject'}", flush=True)
        return ok

    @staticmethod
    def _same_code(a: dict, b: dict) -> bool:
        sa = (a.get("code_object") or {}).get("isa_sha")
        return bool(sa) and sa == (b.get("code_object") or {}).get("isa_sha")

    @staticmethod
    def score(rec: dict) -> float:
        if rec.get("status") != "ok":
            return float("-inf")
        return rec.get("tflops_confirmed", rec.get("tflops", 0.0))

    def _pick(self, results: list[dict], best_rec: dict) -> dict | None:
        """Best result that changes the code, clears the threshold and confirms."""
        ranked = sorted((r for r in results if not self._same_code(r, best_rec)), key=self.score, reverse=True)
        for cand in ranked[:2]:
            if self.score(cand) <= self.score(best_rec) * (1 + self.args.threshold):
                break
            if self._confirm(cand, best_rec):
                return cand
        return None

    def _log(self, rec: dict) -> None:
        pmc = rec.get("pmc", {})
        extra = ""
        if pmc.get("status") == "ok":
            extra = f" mfma_util={pmc.get('mfma_util', 0):.3f} clock={pmc.get('clock_mhz', 0):.0f}MHz"
        print(
            f"[tune r{rec.get('round')}] {rec.get('move')}: status={rec.get('status')} "
            f"tflops={rec.get('tflops', 0):.1f}{extra} ({rec.get('wall_s', 0):.0f}s)",
            flush=True,
        )

    # ------------------------------------------------------------ search

    def run(self) -> dict:
        best = dict(self.defaults)
        if self.args.start:
            best.update(json.loads(self.args.start))
        best_rec = self.evaluate(best, move="start", parent=None, round_=0, pmc=self.args.pmc != "none")
        round_ = 0
        while len(self.records) < self.args.max_evals:
            round_ += 1
            singles = []
            for knob, values in KNOBS.items():
                for value in values:
                    if value != best[knob]:
                        singles.append({**best, knob: value})
            results = [
                self.evaluate(c, move=describe(c, best), parent=best, round_=round_, pmc=self.args.pmc == "all")
                for c in singles
                if len(self.records) < self.args.max_evals or config_key(c) in self.records
            ]
            winner = self._pick(results, best_rec)
            if winner:
                self._accept(winner, best, round_, "single")
                best, best_rec = dict(winner["config"]), winner
                continue
            # Every single move of this round failed on the current best. Rank
            # them by how close they came; pairs of near-misses go first, by
            # the sum of the two ranks.
            failed = []
            for c in sorted(results, key=self.score, reverse=True):
                if self.score(c) == float("-inf") or self._same_code(c, best_rec):
                    continue
                failed += [(k, json.dumps(c["config"][k])) for k in KNOBS if c["config"][k] != best[k]]
            # Look-back: combine moves that failed alone.
            pairs = []
            for i, j in sorted(itertools.combinations(range(len(failed)), 2), key=lambda ij: (sum(ij), ij[1])):
                (k1, v1), (k2, v2) = failed[i], failed[j]
                if k1 != k2:
                    pairs.append({**best, k1: json.loads(v1), k2: json.loads(v2)})
            pairs = pairs[: self.args.max_pairs]
            results = [
                self.evaluate(c, move="lookback " + describe(c, best), parent=best, round_=round_, pmc=self.args.pmc == "all")
                for c in pairs
                if len(self.records) < self.args.max_evals
            ]
            winner = self._pick(results, best_rec)
            if winner:
                self._accept(winner, best, round_, "lookback")
                best, best_rec = dict(winner["config"]), winner
                continue
            break
        if self.args.pmc != "none" and "pmc" not in best_rec:
            best_rec = self.evaluate(best, move="final", parent=None, round_=round_, pmc=True)
        self.write_report(final=best_rec)
        return best_rec

    def _accept(self, winner: dict, best: dict, round_: int, how: str) -> None:
        winner["accepted"] = {"round": round_, "how": how, "from": config_key(best)}
        if self.args.pmc == "best" and "pmc" not in winner:
            winner["pmc"] = self._pmc(winner["config"])
        self._append(winner)
        print(f"[tune r{round_}] ACCEPT ({how}): {describe(winner['config'], self.defaults)}", flush=True)

    # ------------------------------------------------------------ report

    def write_report(self, final: dict | None = None) -> None:
        lines = [
            "# MXFP6 tiled GEMM tuning ledger",
            "",
            f"shape {self.args.shape}, data {self.args.data}. Measured numbers: keep outside the repository.",
            "",
            "| round | move | status | TFLOP/s | MFMA util | clock MHz | HBM GB/s | VGPR | AGPR | spill | accepted |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        seen = set()
        for rec in self.history:
            key = config_key(rec.get("config", {}))
            if key in seen and "accepted" not in rec:
                continue
            seen.add(key)
            pmc = rec.get("pmc", {}) or {}
            co = rec.get("code_object", {}) or {}
            lines.append(
                f"| {rec.get('round', '')} | {rec.get('move', '')} | {rec.get('status')} "
                f"| {rec.get('tflops', 0):.1f} | {pmc.get('mfma_util', float('nan')):.3f} "
                f"| {pmc.get('clock_mhz', float('nan')):.0f} | {pmc.get('hbm_gbps', float('nan')):.0f} "
                f"| {co.get('vgpr_count', '')} | {co.get('agpr_count', '')} | {co.get('vgpr_spill_count', '')} "
                f"| {'yes' if rec.get('accepted') else ''} |"
            )
        if final is not None:
            lines += ["", "## Best", "", "```json", json.dumps(final.get("config"), indent=2), "```"]
        (self.ledger_dir / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ledger", required=True, help="output directory, outside the repository")
    p.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "rocke_mxfp6_cache"))
    p.add_argument("--shape", default="4096,4096,65536")
    p.add_argument("--data", choices=("random", "zero"), default="random")
    p.add_argument("--start", help="JSON knob overrides for the starting configuration")
    p.add_argument("--max-evals", type=int, default=60)
    p.add_argument("--max-pairs", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.01, help="relative gain needed to accept a move")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--sustain-ms", type=int, default=1500, help="load before timing, so the clock has settled")
    p.add_argument("--confirm", type=int, default=2, help="alternating re-timings of winner and incumbent")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--pmc", choices=("none", "best", "all"), default="best")
    args = p.parse_args(argv)
    best = Tuner(args).run()
    print(json.dumps({"best": best.get("config"), "tflops": best.get("tflops")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
