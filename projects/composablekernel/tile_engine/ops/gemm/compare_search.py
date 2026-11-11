#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Prepare paired-seed search ablations and summarize completed driver runs.

This tool never launches GPU jobs. Run each manifest entry with the same driver
arguments, hardware/partition, timing environment and cache policy.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from adaptive_search import SearchSettings, SettingsError, sha256_file, valid_latency  # noqa: E402

# Each arm names its control and the settings patches it applies, in order.
ARMS = {
    "baseline": ("baseline", ()),
    "listed_tpe": ("baseline", ("listed_tpe",)),
    "coupled_tpe": ("baseline", ("coupled_tpe",)),
    "broad_prior": ("baseline", ("prior",)),
    "diverse_warm": ("baseline", ("warm",)),
    "confirmation_control": ("baseline", ("long_confirm",)),
    "race": ("confirmation_control", ("long_confirm", "race")),
    "build_aware": ("baseline", ("build",)),
    "portfolio_control": ("baseline", ("portfolio",)),
    "portfolio_tpe": ("portfolio_control", ("portfolio", "tpe_member")),
    "combined": ("baseline", (
        "prior", "warm", "long_confirm", "race", "build", "portfolio", "tpe_member")),
}


def prepare(baseline, ranked_csv, diverse_pairs, output, seeds):
    baseline = SearchSettings.from_json(baseline)
    if (baseline.strategy != "tpe" or baseline.build_aware is not None
            or baseline.portfolio_members is not None or baseline.confirm.mode != "fixed"
            or baseline.warm_start is None or baseline.warm_start.pairs_file is None
            or baseline.warm_start.use_prior or baseline.tpe_candidate_pool != "sampled"):
        raise SettingsError("baseline must be fixed-confirmation TPE with warm pairs, "
                            "no prior, and no experimental controls")
    if not seeds or len(set(seeds)) != len(seeds):
        raise SettingsError("seeds must be a nonempty distinct list")
    ranked_csv, diverse_pairs = Path(ranked_csv).resolve(), Path(diverse_pairs).resolve()
    if not ranked_csv.is_file() or not diverse_pairs.is_file():
        raise SettingsError("ranked CSV and diverse pairs must be existing files")
    patches = {
        "listed_tpe": {"tpe_candidate_pool": "listed"},
        "coupled_tpe": {"strategy": "coupled_tpe"},
        "prior": {"warm_start": {"ranked_csv": str(ranked_csv), "use_prior": True}},
        "warm": {"warm_start": {"pairs_file": str(diverse_pairs)}},
        "long_confirm": {"confirm": {"fresh_calls": max(9, baseline.confirm.fresh_calls)}},
        "race": {"confirm": {"mode": "race", "min_calls": 3}},
        "build": {"build_aware": {"rank_fraction": 0.1, "explore_probability": 0.2}},
        "portfolio": {"strategy": "portfolio"},
        "tpe_member": {"portfolio_members": ["hill", "gp", "de", "tpe"]},
    }
    output = Path(output).resolve()
    planned = []
    for seed in seeds:
        cases = []
        for arm in ARMS:
            data = copy.deepcopy(baseline.to_json())
            data["seed"] = seed
            for patch in ARMS[arm][1]:
                for key, value in patches[patch].items():
                    if isinstance(value, dict):
                        data.setdefault(key, {}).update(value)
                    else:
                        data[key] = value
            validated = SearchSettings.from_json(data).to_json()
            name = f"{arm}-seed-{seed}"
            cases.append({
                "arm": arm, "seed": seed,
                "config": str(output / "configs" / f"{name}.json"),
                "output_dir": str(output / "runs" / name),
                "settings": validated,
                "warm_start_sha256": {
                    validated["warm_start"][key]: sha256_file(validated["warm_start"][key])
                    for key in ("pairs_file", "ranked_csv") if key in validated["warm_start"]
                },
            })
        random.Random(seed).shuffle(cases)
        planned.extend(cases)
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    (output / "runs").mkdir()
    for case in planned:
        Path(case["config"]).write_text(json.dumps(case["settings"], indent=2) + "\n")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "runs": planned}, indent=2) + "\n")
    return manifest


def _setup_failed(meta, summary):
    # Only the coordinator's explicit minimal failure record bypasses validation.
    # Incomplete/malformed successful-setup records must still fail loudly.
    return (bool(meta.get("error"))
            and not {"context", "problems", "warm_start_sha256"} & set(meta)
            and summary["status"] == "no_winner" and summary["exit_code"] != 0
            and not summary["winners"])


def _read_run(case):
    directory = Path(case["output_dir"])
    paths = [directory / name for name in ("run.json", "summary.json")]
    if not all(p.is_file() for p in paths):
        return None
    meta, summary = [json.loads(p.read_text()) for p in paths]
    if meta["settings"] != case["settings"]:
        raise ValueError(f"{directory}: settings differ from the prepared comparison")
    if _setup_failed(meta, summary):
        return meta, summary
    if meta.get("warm_start_sha256") != case["warm_start_sha256"]:
        raise ValueError(f"{directory}: warm/prior inputs changed after comparison preparation")
    if case["settings"]["warm_start"].get("use_prior") and any(
        p["strategy"] != "exhaustive" and p["prior_entries"] != p["eligible"]
        for p in meta["problems"].values()
    ):
        raise ValueError(f"{directory}: the prior does not cover every eligible candidate")
    for winner in summary["winners"].values():
        if not valid_latency(winner["median_ms"]):
            raise ValueError(f"{directory}: invalid confirmed latency")
    return meta, summary


def _check_comparable(a, b, directory):
    ca, cb = a["context"], b["context"]
    fields = ("arch", "dtype", "layout", "variant", "domain_sha256", "devices",
              "timing_environment", "build", "sources")
    differences = [k for k in fields if ca.get(k) != cb.get(k)]
    if ca.get("domain_sha256") is None:
        differences.append("missing domain_sha256")
    differences.extend("measurement." + k
                       for k in ("verify", "verify_tol", "kernel_timeout", "batch_size")
                       if ca.get("measurement", {}).get(k) != cb.get("measurement", {}).get(k))
    # Enabled caches have a fresh per-run directory (and may have a free-space
    # derived byte cap). Compare the mode, not those runtime allocation details.
    if ((ca.get("measurement", {}).get("host_cache") is None)
            != (cb.get("measurement", {}).get("host_cache") is None)):
        differences.append("measurement.host_cache mode")
    if set(a["problems"]) != set(b["problems"]):
        differences.append("problems")
    else:
        differences.extend(f"warm_start count for {p}"
                           for p in a["problems"]
                           if len(a["problems"][p]["warm_start"])
                           != len(b["problems"][p]["warm_start"]))
    if differences:
        raise ValueError(f"{directory}: control differs in {', '.join(differences)}; "
                         "do not pool these runs")


def summarize(manifest, stream):
    cases = json.loads(Path(manifest).read_text())["runs"]
    runs = {(c["arm"], c["seed"]): _read_run(c) for c in cases}
    fields = ("arm", "seed", "control_arm", "status", "exit_code", "error", "winners", "problems",
              "exhaustive_problems", "paired_winners", "same_winners", "control_relative_quality",
              "builds", "measurements", "confirmation_calls",
              "wall_s", "ask_s", "build_s", "measure_s", "confirm_s", "ck_commit")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    complete = True
    for case in sorted(cases, key=lambda c: (c["seed"], list(ARMS).index(c["arm"]))):
        row = dict.fromkeys(fields, "")
        control = ARMS[case["arm"]][0]
        row.update(arm=case["arm"], seed=case["seed"], control_arm=control)
        current = runs[(case["arm"], case["seed"])]
        if current is None:
            row["status"] = "missing"
            complete = False
        else:
            meta, summary = current
            row.update(status=summary["status"], exit_code=summary["exit_code"],
                       error=summary["error"], winners=len(summary["winners"]))
            if _setup_failed(meta, summary):
                row.update(status="setup_failed", error=summary["error"] or meta["error"])
                complete = False
                writer.writerow(row)
                continue
            baseline = runs.get((control, case["seed"]))
            n = len(meta["problems"])
            row.update(problems=n, builds=summary["budget"]["builds_charged"],
                       measurements=summary["counts"]["measured"],
                       confirmation_calls=summary["counts"]["confirm_calls"],
                       wall_s=summary["wall_s"],
                       exhaustive_problems=sum(p["strategy"] == "exhaustive"
                                               for p in meta["problems"].values()),
                       ck_commit=meta["context"].get("sources", {}).get("ck_commit", ""))
            for label, key in (("ask", "ask"), ("build", "build_missing"),
                               ("measure", "measure_pairs"), ("confirm", "confirm")):
                row[label + "_s"] = summary.get("phase_s", {}).get(key, "")
            complete &= summary["status"] == "complete" and summary["exit_code"] == 0
            if baseline is not None and not _setup_failed(*baseline):
                base_meta, base_summary = baseline
                _check_comparable(meta, base_meta, case["output_dir"])
                paired = set(summary["winners"]) & set(base_summary["winners"])
                row["paired_winners"] = len(paired)
                row["same_winners"] = sum(summary["winners"][p]["kernel"]
                                          == base_summary["winners"][p]["kernel"]
                                          for p in paired)
                if n and len(paired) == n:
                    row["control_relative_quality"] = math.exp(sum(
                        math.log(base_summary["winners"][p]["median_ms"])
                        - math.log(summary["winners"][p]["median_ms"])
                        for p in sorted(paired)
                    ) / n)
        writer.writerow(row)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Write eleven paired ablations per seed")
    prep.add_argument("--baseline", required=True)
    prep.add_argument("--ranked-csv", required=True, help="Full eligible-domain predictor CSV")
    prep.add_argument("--diverse-pairs", required=True)
    prep.add_argument("--output", required=True, help="New comparison directory")
    prep.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    report = commands.add_parser("summarize", help="Write per-run CSV to stdout")
    report.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(prepare(args.baseline, args.ranked_csv, args.diverse_pairs,
                          args.output, args.seeds))
            return 0
        return summarize(args.manifest, sys.stdout)
    except (OSError, ValueError, KeyError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    sys.exit(main())
