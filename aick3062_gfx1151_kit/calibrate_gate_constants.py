#!/usr/bin/env python3
"""Calibrate the backward bench's gate constants for one RDNA device from its own
sweep records (AICK-3062). The shipped values are private and unavailable, so the
thresholds are derived from measured noise on the device the tables are tuned on:

* BAND_K       = 1.25 x max over A/A twin pairs of |rho - 1| / max(spread):
                 identical kernels (an arm and its ``#aa`` twin) never ``fail``.
* TIE_EPS      = max(0.005, 1.25 x max A/A |rho - 1|): the final tie covers twin noise,
                 never tighter than 0.5 %.
* NOISY_SPREAD = p99 of per-arm spreads (relative IQR of attempt medians), rounded up
                 to 0.01: only the noisiest ~1 % of cells get more attempts.
* RERUN_MAX = 3, ATTEMPTS_NOISY = 9, CLIFF_BAND = 0.3: procedural counts (the bench's
                 rule-test values); CLIFF_BAND is not read by the sweep / confirm path.

Usage: calibrate_gate_constants.py <arch>  -> writes gate_constants_<arch>.json next to
$AICK_WORK (outside the source tree) with the values and their provenance.
"""

import glob
import json
import math
import os
import sys
from pathlib import Path

# work directory (outside the source tree) holding sweep/<arch>/ and receiving the output
HERE = Path(os.environ.get("AICK_WORK", Path(__file__).resolve().parent)).resolve()


def records(arch):
    pat = (
        HERE
        / "sweep"
        / arch
        / f"{arch}-general-d*"
        / "*"
        / "timing-*"
        / "group-*"
        / "cells.jsonl"
    )
    for p in glob.glob(str(pat)):
        for line in open(p, encoding="utf-8"):
            yield json.loads(line)


def quantile(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def calibrate(arch):
    spreads, aa_dev, aa_norm = [], [], []
    for r in records(arch):
        arms = r.get("arms", {})
        for name, a in arms.items():
            if a.get("status") == "ok" and a.get("spread") is not None:
                spreads.append(a["spread"])
            base = arms.get(name[:-3]) if name.endswith("#aa") else None
            if base and a.get("status") == "ok" and base.get("status") == "ok":
                rho = a["median_ms"] / base["median_ms"]
                s = max(a["spread"], base["spread"], 1e-9)
                aa_dev.append(abs(rho - 1.0))
                aa_norm.append(abs(rho - 1.0) / s)
    if len(aa_dev) < 20 or len(spreads) < 100:
        raise SystemExit(
            f"{arch}: too little data (A/A pairs {len(aa_dev)}, arms {len(spreads)})"
        )
    consts = {
        "BAND_K": math.ceil(1.25 * max(aa_norm) * 2) / 2,  # round up to 0.5
        "TIE_EPS": max(0.005, math.ceil(1.25 * max(aa_dev) * 1000) / 1000),
        "NOISY_SPREAD": math.ceil(quantile(spreads, 0.99) * 100) / 100,
        "RERUN_MAX": 3,
        "ATTEMPTS_NOISY": 9,
        "CLIFF_BAND": 0.3,
    }
    provenance = {
        "arch": arch,
        "method": __doc__.strip().splitlines()[0],
        "arms": len(spreads),
        "aa_pairs": len(aa_dev),
        "spread_p50": quantile(spreads, 0.5),
        "spread_p99": quantile(spreads, 0.99),
        "aa_abs_max": max(aa_dev),
        "aa_norm_max": max(aa_norm),
    }
    return consts, provenance


def main():
    arch = sys.argv[1]
    consts, prov = calibrate(arch)
    out = HERE / f"gate_constants_{arch}.json"
    out.write_text(
        json.dumps({"constants": consts, "provenance": prov}, indent=2) + "\n"
    )
    os.chmod(out, 0o600)
    print(json.dumps({"constants": consts, "provenance": prov}, indent=2))


if __name__ == "__main__":
    main()
