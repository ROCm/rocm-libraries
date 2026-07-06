# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Command-line entry point for the user tool (dev OR agent).

`python -m rocke.benchmark.perf.tool <cmd>` ties the primitives + store + self-check together:

  profile    -- measure a kernel command N times, aggregate, store, self-check
               vs the prior stored run for that (arch, kernel, shape).   [GPU]
  occupancy  -- read a compiled HSACO's ELF notes -> resources/occupancy. [no GPU]
  compare    -- load the stored history and report improve/regress per kernel.
               [no GPU]

Every command takes `--json` so an agent can consume structured output instead of
the human text. Only this layer writes (via `store`, to the user cache dir).
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from rocke.benchmark.perf import aggregate as _aggregate
from rocke.benchmark.perf import harness as _harness
from rocke.benchmark.perf import occupancy as _occupancy
from rocke.benchmark.perf import report as _report
from rocke.benchmark.perf import schema as _schema
from . import selfcheck as _selfcheck
from . import store as _store


def _emit(obj, *, as_json: bool, human: str) -> None:
    print(json.dumps(obj, sort_keys=True, indent=2) if as_json else human)


def _shape_arg(text: Optional[str]) -> dict:
    if not text:
        return {}
    try:
        val = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--shape must be JSON: {e}")
    if not isinstance(val, dict):
        raise SystemExit("--shape must be a JSON object, e.g. '{\"M\":512}'")
    return val


def _cmd_profile(a: argparse.Namespace) -> int:
    # argparse.REMAINDER keeps the literal '--' separator as cmd[0]; drop it so we
    # don't try to exec a program named '--'.
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("profile: give the kernel command after '--'")
    shape = _shape_arg(a.shape)
    _warn = lambda m: print(f"warning: {m}", file=sys.stderr)
    samples = []
    for _ in range(max(1, a.repeats)):
        samples.append(_harness.profile(
            cmd, a.arch, match=a.match_kernel, label=a.kernel_name,
            op=a.op, shape=shape, warn=_warn))
    rec = _aggregate.aggregate(samples)

    identity = _schema.identity(rec)
    prior = _store.records_for(_store.load(cache=a.cache), identity)
    if prior:
        result = _selfcheck.compare(prior[-1], rec,
                                    threshold=a.threshold, noise_k=a.noise_k)
    else:
        result = {"verdict": "no_baseline", "n_runs": len(prior),
                  "identity": {"arch": identity[0], "kernel_name": identity[1],
                               "shape": identity[2]}}
    if not a.no_store:
        _store.append(rec, cache=a.cache)

    _emit({"record": rec, "selfcheck": result}, as_json=a.json,
          human=_report.format_record(rec) + "\n" + _selfcheck.format_result(result))
    return 1 if result.get("verdict") == "regressed" else 0


def _cmd_occupancy(a: argparse.Namespace) -> int:
    try:
        with open(a.hsaco, "rb") as f:
            data = f.read()
    except OSError as e:
        raise SystemExit(f"occupancy: cannot read {a.hsaco}: {e}")
    res = _occupancy.resources(data, a.arch)
    if not res:
        raise SystemExit("occupancy: could not read ELF notes "
                         "(need llvm-readelf and a valid HSACO)")
    human = "\n".join(f"{k}: {v}" for k, v in res.items())
    _emit(res, as_json=a.json, human=human)
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    records = _store.load(cache=a.cache)
    if a.all:
        identities = sorted(_store.group_by_identity(records))
    else:
        if not (a.arch and a.kernel_name):
            raise SystemExit("compare: give --arch and --kernel-name, or --all")
        identities = [(a.arch, a.kernel_name, _schema.shape_signature(_shape_arg(a.shape)))]

    results = [_selfcheck.check_history(records, ident,
                                        threshold=a.threshold, noise_k=a.noise_k)
               for ident in identities]
    regressed = any(r.get("verdict") == "regressed" for r in results)
    _emit(results, as_json=a.json,
          human="\n".join(_selfcheck.format_result(r) for r in results) or "(no history)")
    return 1 if regressed else 0


def _build_parser() -> argparse.ArgumentParser:
    # --json / --cache accepted either before OR after the subcommand. SUPPRESS
    # defaults so a value given at one position isn't clobbered by the other;
    # main() fills the real defaults.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="emit JSON (for agents)")
    common.add_argument("--cache", default=argparse.SUPPRESS,
                        help="override the cache dir")

    p = argparse.ArgumentParser(prog="rocke.benchmark.perf.tool", parents=[common])
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("profile", parents=[common],
                        help="measure -> aggregate -> store -> self-check")
    pr.add_argument("--arch", required=True)
    pr.add_argument("--op", default="unknown")
    pr.add_argument("--shape", default=None, help="JSON object, e.g. '{\"M\":512}'")
    pr.add_argument("--kernel-name", dest="kernel_name", default=None,
                    help="identity name for pairing (stable across edits that "
                         "rename the dispatched symbol); default = dispatched symbol")
    pr.add_argument("--match-kernel", dest="match_kernel", default=None,
                    help="substring of the dispatched symbol to profile; "
                         "default = busiest non-helper dispatch")
    pr.add_argument("--repeats", type=int, default=1)
    pr.add_argument("--threshold", type=float, default=_selfcheck.DEFAULT_THRESHOLD)
    pr.add_argument("--noise-k", dest="noise_k", type=float,
                    default=_selfcheck.DEFAULT_NOISE_K)
    pr.add_argument("--no-store", dest="no_store", action="store_true")
    pr.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <kernel launch argv>")
    pr.set_defaults(func=_cmd_profile)

    oc = sub.add_parser("occupancy", parents=[common],
                        help="ELF-note resources for an HSACO (no GPU)")
    oc.add_argument("hsaco")
    oc.add_argument("--arch", required=True)
    oc.set_defaults(func=_cmd_occupancy)

    cm = sub.add_parser("compare", parents=[common],
                        help="improve/regress from stored history (no GPU)")
    cm.add_argument("--arch", default=None)
    cm.add_argument("--kernel-name", dest="kernel_name", default=None)
    cm.add_argument("--shape", default=None, help="JSON object, e.g. '{\"M\":512}'")
    cm.add_argument("--all", action="store_true", help="every identity in history")
    cm.add_argument("--threshold", type=float, default=_selfcheck.DEFAULT_THRESHOLD)
    cm.add_argument("--noise-k", dest="noise_k", type=float,
                    default=_selfcheck.DEFAULT_NOISE_K)
    cm.set_defaults(func=_cmd_compare)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    # SUPPRESS means these may be absent; fill real defaults once, centrally.
    args.json = getattr(args, "json", False)
    args.cache = getattr(args, "cache", None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
